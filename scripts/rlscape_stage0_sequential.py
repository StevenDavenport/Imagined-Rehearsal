#!/usr/bin/env python3
"""Canonical entry point for RLScape Stage 0 sequential experiments.

The implementation remains in the historical M3 module because completed
checkpoint manifests import helpers from that path. This stable entry point
separates the scientific Stage 0 name from the old engineering milestone.
"""

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rlscape_m3_sequential import main


if __name__ == '__main__':
  raise SystemExit(main())
