#!/usr/bin/env python3
"""Legacy CLI alias → ``order_invariance_certificate_k10`` (same flags)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

if __name__ == "__main__":
    import order_invariance_certificate_k10 as _oi

    _oi.main()
