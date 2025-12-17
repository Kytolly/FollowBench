# src/Ego2ExoFollowShot/cli/main.py
import click
from ..bench.runner import run_evaluation
from ..competition.validator import validate_submission

@click.group()
def cli():
    """Ego2Exo Benchmark Command Line Tool"""
    pass

@cli.command()
@click.option('--submission', required=True, help='Path to submission folder/json')
@click.option('--output', default='./results', help='Output directory')
def evaluate(submission, output):
    """Run full evaluation on a submission"""
    run_evaluation(submission, output)

@cli.command()
@click.option('--submission', required=True, help='Path to submission folder/json')
def validate(submission):
    """Check if submission format is valid"""
    if validate_submission(submission):
        click.echo("✅ Submission is valid!")
    else:
        click.echo("❌ Submission is invalid.")
        exit(1)

if __name__ == '__main__':
    cli()