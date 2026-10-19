"""Single Vercel entrypoint for all FastAPI routes.

Keeping one entrypoint is important for the PyTorch deployment: Vercel can
package the shared runtime and model once instead of making three copies for
analyze.py, health.py and models.py.
"""
from backend.app import app
