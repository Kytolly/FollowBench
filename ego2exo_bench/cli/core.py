import click

from .. import Bench 
from ..dataflow.submission import Submission 
# 检查提交文件格式是否合法
# python -m src.cli.core validate \
#        --submission ./cache/gen/submission.json \
#        --source ./assets/Test
# 运行完整的 Benchmark 流程
# 基础运行
# python -m src.cli.core evaluate 
#        --submission ./cache/gen/submission.json
# 指定输出目录和资源目录
# python -m src.cli.core evaluate \
#        --submission ./cache/gen/submission.json \
#        --output ./results/my_experiment \
#        --assets ./assets/Test

@click.group()
def cli():
    """Benchmark Command Line Tool"""
    pass

@cli.command()
@click.option('--submission', required=True, help='Path to submission.json')
@click.option('--output', default='./output', help='Output directory')
@click.option('--assets', default='./assets', help='Assets directory')
def evaluate(submission, output, assets):
    """Run full evaluation on a submission"""
    bench = Bench(device='cuda', assets_root=assets)
    bench.evaluate(submission_path=submission, output_dir=output)

@cli.command()
@click.option('--submission', required=True, help='Path to submission folder/json')
@click.option('--source', required=True, help='Path to video source root')
def validate(submission, source):
    """Check if submission format is valid"""
    sub = Submission(submission_path=submission, source_path=source)
    try:
        submission.validate_all()
        click.echo(f"✅ Submission is valid! Found {len(sub)} cases.")
    except Exception as e:
        click.echo(f"❌ Submission is invalid: {e}")
        exit(1)

if __name__ == '__main__':
    cli()