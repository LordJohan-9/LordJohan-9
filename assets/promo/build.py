"""Regenerate the training trace and all profile artwork."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    for script in ("train_trace.py", "generate.py"):
        subprocess.run([sys.executable, str(HERE / script)], cwd=HERE, check=True)


if __name__ == "__main__":
    main()
