"""
Script: Run analysis
====================
Expose the analysis CLI from a source checkout.

Requirements
------------
- Add the repository src directory to the Python import path.
- Forward command-line arguments and the exit code to ir_system.cli.analysis.
- Keep configuration and data loading in the CLI; do not duplicate either here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_system.cli.analysis import main


if __name__ == "__main__":
    raise SystemExit(main())
