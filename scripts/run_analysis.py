"""Run the analysis CLI from a source checkout without installing the package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_system.cli.analysis import main


if __name__ == "__main__":
    raise SystemExit(main())
