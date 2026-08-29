"""Browser-based tool-calling evaluation workbench.

The interactive /api/run endpoint is an SSE stream. Besides the final
OpenAI-compatible result it reports inference progress so the browser can
show whether the model is doing prompt prefill or autoregressive forward
propagation, together with decode TPS.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

# The Needle checkout is a repository submodule. Resolve it from the project
# root so callers no longer need a machine-specific --needle-source path.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_NEEDLE_SOURCE = os.path.join(_PROJECT_ROOT, "needle")

