"""Command-line interface for the EgoExo Translation Benchmark.

This module provides CLI commands for evaluating submissions and validating
submission formats using the Click framework.

Example usage:
    # Validate submission format
    python -m egoexo_translation_bench.cli.core validate \
           --submission ./cache/gen/submission.json \
           --source ./assets/Test

    # Run full benchmark evaluation
    python -m egoexo_translation_bench.cli.core evaluate \
           --submission ./cache/gen/submission.json \
           --output ./results/my_experiment \
           --assets ./assets/Test
"""

from typing import Optional
import sys
import click

from .. import Bench 
from ..dataflow.submission import Submission 


@click.group()
def cli() -> None:
    """EgoExo Translation Benchmark Command Line Tool.
    
    This tool provides commands for validating submissions and running
    benchmark evaluations on ego-to-exocentric video translation models.
    """
    pass


@cli.command()
@click.option(
    '--submission', 
    required=True, 
    type=click.Path(exists=True),
    help='Path to submission.json file containing model results'
)
@click.option(
    '--output', 
    default='./output', 
    type=click.Path(),
    help='Output directory for evaluation results (default: ./output)'
)
@click.option(
    '--assets', 
    default='./assets', 
    type=click.Path(exists=True),
    help='Assets directory containing test data (default: ./assets)'
)
@click.option(
    '--device',
    default='cuda',
    type=click.Choice(['cuda', 'cpu']),
    help='Device to run evaluation on (default: cuda)'
)
def evaluate(submission: str, output: str, assets: str, device: str) -> None:
    """Run full evaluation on a submission.
    
    This command loads a submission file, runs all benchmark metrics,
    and generates a comprehensive evaluation report.
    
    Args:
        submission: Path to the submission JSON file
        output: Directory to save evaluation results
        assets: Directory containing benchmark assets and test data
        device: Computing device ('cuda' or 'cpu')
    """
    try:
        bench = Bench(device=device, assets_root=assets)
        # Note: The Bench.evaluate method expects a Submission object, not a path
        sub = Submission(submission_path=submission, source_path=assets)
        bench.evaluate(submission=sub, output_dir=output)
        click.echo(f"✅ Evaluation completed successfully! Results saved to {output}")
    except Exception as e:
        click.echo(f"❌ Evaluation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--submission', 
    required=True, 
    type=click.Path(exists=True),
    help='Path to submission JSON file or directory'
)
@click.option(
    '--source', 
    required=True, 
    type=click.Path(exists=True),
    help='Path to video source root directory'
)
def validate(submission: str, source: str) -> None:
    """Check if submission format is valid.
    
    This command validates the submission file format, checks for required
    metadata, verifies video file existence and properties, and ensures
    compliance with benchmark requirements.
    
    Args:
        submission: Path to the submission file or directory
        source: Path to the root directory containing generated videos
    """
    try:
        sub = Submission(submission_path=submission, source_path=source)
        sub.validate_all()
        click.echo(f"✅ Submission is valid! Found {len(sub)} cases.")
    except Exception as e:
        click.echo(f"❌ Submission is invalid: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
