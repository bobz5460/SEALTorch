from pathlib import Path
import sys
directory = Path(__file__).resolve().parent
sys.path.insert(0, str(directory.parents[2]))
from webui.graph_lab import render_saved
render_saved(directory)
