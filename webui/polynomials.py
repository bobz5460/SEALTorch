"""Shared activation-polynomial definitions for plaintext and FIDESlib runs."""
from __future__ import annotations

import math

import numpy as np

METHODS = ("chebyshev", "taylor")
RANGES = (1.0, 2.0, 4.0, 8.0)
PLAINTEXT_DEGREES = tuple(range(1, 21))
ENCRYPTED_DEGREES = tuple(range(1, 10))


def activation_value(name: str, values):
    values = np.asarray(values, dtype=np.float64)
    if name == "relu":
        return np.maximum(values, 0.0)
    if name == "tanh":
        return np.tanh(values)
    if name == "gelu":
        erf = np.vectorize(math.erf, otypes=[float])
        return 0.5 * values * (1.0 + erf(values / math.sqrt(2.0)))
    raise ValueError(f"unsupported activation: {name}")


def taylor_power_coefficients(name: str, degree: int) -> np.ndarray:
    if name == "relu":
        raise ValueError("Taylor approximation is undefined for ReLU at zero")
    coefficients = np.zeros(degree + 1, dtype=np.float64)
    if name == "tanh":
        known = {
            1: 1.0, 3: -1 / 3, 5: 2 / 15, 7: -17 / 315,
            9: 62 / 2835, 11: -1382 / 155925,
            13: 21844 / 6081075, 15: -929569 / 638512875,
            17: 6404582 / 10854718875, 19: -443861162 / 1856156927625,
        }
        for order, value in known.items():
            if order <= degree:
                coefficients[order] = value
        return coefficients
    if name == "gelu":
        if degree >= 1:
            coefficients[1] = 0.5
        for n in range(0, 10):
            order = 2 * n + 2
            if order > degree:
                break
            coefficients[order] = ((-1.0) ** n /
                (math.sqrt(2.0 * math.pi) * (2.0 ** n) * math.factorial(n) * (2 * n + 1)))
        return coefficients
    raise ValueError(f"unsupported activation: {name}")


def coefficients(name: str, method: str, degree: int, interval: float) -> np.ndarray:
    """Return Chebyshev-basis coefficients over ``[-interval, interval]``."""
    if method not in METHODS:
        raise ValueError(f"unsupported approximation method: {method}")
    if degree < 1 or degree > 20:
        raise ValueError("degree must be in [1, 20]")
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("range must be positive and finite")
    if method == "chebyshev":
        return np.polynomial.chebyshev.chebinterpolate(
            lambda normalized: activation_value(name, normalized * interval), degree)
    power = taylor_power_coefficients(name, degree)
    nodes = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))
    values = np.polynomial.polynomial.polyval(nodes * interval, power)
    return np.polynomial.chebyshev.chebfit(nodes, values, degree)


def evaluate(values, chebyshev_coefficients, interval: float):
    return np.polynomial.chebyshev.chebval(
        np.asarray(values, dtype=np.float64) / interval,
        np.asarray(chebyshev_coefficients, dtype=np.float64),
    )


def fideslib_coefficients(chebyshev_coefficients) -> np.ndarray:
    """Convert NumPy's Chebyshev convention to OpenFHE/FIDESlib's c0/2 convention."""
    result = np.asarray(chebyshev_coefficients, dtype=np.float64).copy()
    if result.size:
        result[0] *= 2.0
    return result


def approximation_error(name: str, chebyshev_coefficients, interval: float,
                        sample_count: int = 10_001) -> dict[str, float]:
    values = np.linspace(-interval, interval, sample_count)
    errors = evaluate(values, chebyshev_coefficients, interval) - activation_value(name, values)
    return {
        "approximation_rmse": float(np.sqrt(np.mean(errors * errors))),
        "approximation_max_error": float(np.max(np.abs(errors))),
    }


def multiplicative_depth(degree: int) -> int:
    """Conservative depth estimate for FIDESlib's Paterson–Stockmeyer evaluator."""
    return 1 if degree <= 1 else math.ceil(math.log2(degree)) + 1
