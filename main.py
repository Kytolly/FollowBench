import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from egoexo_translation_bench.cli.core import cli

if __name__ == '__main__':
    cli()