"""Small local research dashboard for SEALTorch sweeps and drawing checks."""
from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
import threading
import time
from urllib.parse import unquote, urlparse

import numpy as np
import torch

try:
    from .experiments import ExperimentManager, NativeWorker, profile
    from .graph_lab import available_runs, render as render_graph, saved_graphs, schema as graph_schema
    from .models import discover_models, load_model, polynomial_model, prepare_drawing, resolve_model
    from .polynomials import (ENCRYPTED_DEGREES, METHODS, PLAINTEXT_DEGREES, RANGES,
                              coefficients, fideslib_coefficients)
except ImportError:
    from experiments import ExperimentManager, NativeWorker, profile
    from graph_lab import available_runs, render as render_graph, saved_graphs, schema as graph_schema
    from models import discover_models, load_model, polynomial_model, prepare_drawing, resolve_model
    from polynomials import (ENCRYPTED_DEGREES, METHODS, PLAINTEXT_DEGREES, RANGES,
                             coefficients, fideslib_coefficients)

HERE = Path(__file__).resolve().parent
MANAGER = ExperimentManager()
DRAWING_LOCK = threading.Lock()
DRAWING_WORKER: NativeWorker | None = None


def prediction(request: dict) -> dict:
    global DRAWING_WORKER
    path = resolve_model(request["model"])
    model, document = load_model(path)
    image = prepare_drawing(request["pixels"], document)
    mode = request.get("mode", "original")
    interval = float(request.get("range", 2.0)); degree = int(request.get("degree", 3))
    method = request.get("method", "chebyshev")
    coeff = None if mode == "original" else coefficients(document["activation"], method, degree, interval)
    if mode in ("original", "plaintext"):
        if coeff is not None: model = polynomial_model(model, coeff, interval)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"; model.to(device).eval()
        start = time.perf_counter()
        with torch.inference_mode(): logits = model(image.unsqueeze(0).to(device))[0].cpu().tolist()
        if device.startswith("cuda"): torch.cuda.synchronize()
        return {"logits": logits, "prediction": int(np.argmax(logits)),
                "latency_ms": (time.perf_counter() - start) * 1000, "mode": mode}
    if mode != "encrypted": raise ValueError("mode must be original, plaintext, or encrypted")
    he = profile(document, degree, request.get("he_parameters"))
    if not he["feasible"]: raise ValueError(he["reason"])
    native_request = {"model_path": str(path), "pixels": image.reshape(-1).tolist(),
                      "coefficients": fideslib_coefficients(coeff).tolist(),
                      "lower_bound": -interval, "upper_bound": interval,
                      **{key: he[key] for key in ("ring_dim", "depth", "scaling_mod_bits", "first_mod_bits", "device")}}
    with DRAWING_LOCK:
        DRAWING_WORKER = DRAWING_WORKER or NativeWorker()
        result = DRAWING_WORKER.predict(native_request)
    result.update({"prediction": int(np.argmax(result["logits"])), "mode": mode, "he_profile": he})
    return result


class Handler(BaseHTTPRequestHandler):
    server_version = "SEALTorch/0.2"

    def json_response(self, value, status=HTTPStatus.OK):
        body = json.dumps(value).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def body(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/": return self.file(HERE / "index.html")
            if path == "/api/models": return self.json_response({"models": discover_models()})
            if path == "/api/config": return self.json_response({"methods": METHODS, "ranges": RANGES,
                "plaintext_degrees": PLAINTEXT_DEGREES, "encrypted_degrees": ENCRYPTED_DEGREES,
                "gpu_count": max(1, torch.cuda.device_count())})
            if path == "/api/graph/runs": return self.json_response({
                "runs": available_runs(MANAGER.results), "graphs": saved_graphs(MANAGER.results)})
            if path == "/api/runs": return self.json_response({
                "runs": [run.public() for run in reversed(MANAGER.runs.values())],
                "sweeps": [sweep.public(MANAGER.runs) for sweep in reversed(MANAGER.sweeps.values())]})
            if path.startswith("/api/runs/"):
                run_id = path.rsplit("/", 1)[-1]; return self.json_response(MANAGER.runs[run_id].public())
            if path.startswith("/results/"):
                relative = Path(unquote(path.removeprefix("/results/")))
                target = (MANAGER.results / relative).resolve()
                if MANAGER.results.resolve() not in target.parents: raise ValueError("invalid result path")
                return self.file(target)
            self.send_error(HTTPStatus.NOT_FOUND)
        except KeyError: self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as error: self.json_response({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/sweeps":
                sweep = MANAGER.start_sweep(self.body())
                return self.json_response(sweep.public(MANAGER.runs), HTTPStatus.ACCEPTED)
            if path == "/api/benchmarks":
                benchmark = MANAGER.start_benchmark(self.body())
                return self.json_response(benchmark.public(MANAGER.runs), HTTPStatus.ACCEPTED)
            if path == "/api/manual-batches":
                batch = MANAGER.start_manual_batch(self.body())
                return self.json_response(batch.public(MANAGER.runs), HTTPStatus.ACCEPTED)
            if path == "/api/manual-recommendation":
                return self.json_response(MANAGER.recommend_manual_parameters(self.body()))
            if path == "/api/manual-degree-profiles":
                return self.json_response(MANAGER.manual_degree_profiles(self.body()))
            if path == "/api/graph/schema": return self.json_response(graph_schema(MANAGER.results, self.body()))
            if path == "/api/graph/render":
                request = self.body()
                with MANAGER.plot_lock:
                    graph = render_graph(MANAGER.results, request)
                return self.json_response(graph, HTTPStatus.CREATED)
            if path == "/api/predict": return self.json_response(prediction(self.body()))
            if path.startswith("/api/runs/") and path.endswith("/resume"):
                return self.json_response(MANAGER.resume(path.split("/")[3]).public(), HTTPStatus.ACCEPTED)
            if path.startswith("/api/runs/") and path.endswith("/cancel"):
                MANAGER.cancel(path.split("/")[3]); return self.json_response({"cancelled": True})
            if path.startswith("/api/sweeps/") and path.endswith("/cancel"):
                MANAGER.cancel_sweep(path.split("/")[3]); return self.json_response({"cancelled": True})
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as error: self.json_response({"error": str(error)}, HTTPStatus.CONFLICT)

    def do_DELETE(self):
        path = urlparse(self.path).path
        try:
            if path.startswith("/api/runs/"):
                MANAGER.delete_run(path.split("/")[3])
                return self.json_response({"deleted": True})
            if path.startswith("/api/sweeps/"):
                MANAGER.delete_sweep(path.split("/")[3])
                return self.json_response({"deleted": True})
            self.send_error(HTTPStatus.NOT_FOUND)
        except KeyError:
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as error:
            self.json_response({"error": str(error)}, HTTPStatus.CONFLICT)

    def file(self, path: Path):
        if not path.is_file(): return self.send_error(HTTPStatus.NOT_FOUND)
        body = path.read_bytes(); self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(); server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SEALTorch dashboard: http://{args.host}:{args.port}")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        if DRAWING_WORKER: DRAWING_WORKER.close()


if __name__ == "__main__": main()
