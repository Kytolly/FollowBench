"""Main entry point for the EgoExo Translation Benchmark.

This module provides the command-line interface entry point for the benchmark system.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path for module imports
current_dir: Path = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from follow_bench.cli.core import cli


def main() -> None:
    """Main entry point for the CLI application."""
    cli()


if __name__ == '__main__':
    main()
