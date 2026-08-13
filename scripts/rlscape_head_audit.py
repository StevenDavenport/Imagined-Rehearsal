#!/usr/bin/env python3
"""Compatibility entry point for canonical RLScape Stage 1A.

New commands should use ``scripts/rlscape_stage1a_head_audit.py``. This shim is
retained so completed command manifests and older operational notes remain
executable.
"""

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rlscape_stage1a_head_audit import main


if __name__ == '__main__':
  raise SystemExit(main())
