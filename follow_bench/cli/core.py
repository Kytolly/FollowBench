from typing import Optional
import sys
import click
from click import Context

from ..configs import BaseEnvConfig

# =====================================================================
# 1. 定义共用的 CLI 选项组 (利用 click 的回调机制处理全局配置)
# =====================================================================

def build_config_callback(ctx: Context, param, value):
    """
    当所有的命令行参数解析完毕后 Click 会自动执行这个回调。
    在这里我们构建三级配置，并将合并后的 cfg 存入 ctx.obj，供子命令使用。
    """
    p = ctx.params
    
    cfg = BaseEnvConfig()
    cli_overrides = {
        'device': p.get('device'),
        'metrics': list(p.get('metrics')) if p.get('metrics') else None,
        'assets': {
            'path': p.get('assets_path')
        },
        'output': {
            'path': p.get('output_path'),
        },
        'submission': {
            'source_path': p.get('submission_source'),
            'json_path': p.get('submission_json')
        }
    }
    
    config_path = p.get('config')
    if config_path:
        cfg.update_from_yaml(config_path)
    cfg.update_from_dict(cli_overrides)
    
    # 如果指定了 --list，打印配置并直接退出
    if p.get('list_config'):
        click.echo("="*40)
        click.echo("🚀 Current FollowBench Configuration:")
        click.echo(cfg)
        click.echo("="*40)
        sys.exit(0)

    ctx.obj = cfg
    return value


@click.group(invoke_without_command=True)
@click.option('-c', '--config', default='follow_bench/configs/config.yml', type=click.Path(), help='Config file for benchmark running.')
@click.option('--assets_path', default='assets/followbench', type=click.Path(), help='Assets root directory containing dataset.')
@click.option('--output_path', default='./output', type=click.Path(), help='Output directory for evaluation results.')
@click.option('--device', default='cuda', type=click.Choice(['cuda', 'cpu']), help='Device to run evaluation on.')
@click.option('--submission_source', type=click.Path(), help='Path to submission source generated videos root directory.')
@click.option('--submission_json', type=click.Path(), help='Path to submission.json file.')
@click.option('--list', 'list_config', is_flag=True, default=False, help='List all configurations and exit.')
@click.option('--metrics', multiple=True, help='You can use [--metrics mse] [--metrics ssim] etc.')
@click.pass_context
def cli(ctx: Context, config, assets_path, output_path, device, submission_source, submission_json, list_config, metrics) -> None:
    """Benchmark Command Line Tool.
    
    This tool provides commands for validating submissions 
    and running benchmark evaluations on generated videos.
    """
    # 配置构建
    build_config_callback(ctx, None, None)
    
    # 如果用户没有带任何子命令（如 evaluate, validate）且没有带 --list，则打印帮助信息
    if ctx.invoked_subcommand is None and not list_config:
        click.echo(ctx.get_help())

@cli.command()
@click.pass_context
def evaluate(ctx: Context) -> None:
    """Run full evaluation on a submission."""
    cfg: BaseEnvConfig = ctx.obj
    
    if not cfg.submission.json_path or not cfg.submission.source_path:
        click.echo("❌ Error: Missing submission parameters. Please provide --submission_json and --submission_source", err=True)
        sys.exit(1)

    from .. import Bench 
    from ..dataflow.submission import Submission 
    bench = Bench(cfg)
    sub = Submission(submission_path=cfg.submission.json_path, 
                        source_path=cfg.submission.source_path)
    bench.run(cfg, submission=sub)
    click.echo(f"✅ Evaluation completed successfully! Results saved to {cfg.output.path}")

@cli.command()
@click.pass_context
def validate(ctx: Context) -> None:
    """Check if submission format is valid."""
    cfg: BaseEnvConfig = ctx.obj
    
    if not cfg.submission.json_path or not cfg.submission.source_path:
        click.echo("❌ Error: Missing submission parameters.", err=True)
        sys.exit(1)

    try:
        from ..dataflow.submission import Submission 
        sub = Submission(submission_path=cfg.submission.json_path, 
                         source_path=cfg.submission.source_path)
        sub.validate_all()
        click.echo(f"✅ Submission is valid! Found {len(sub)} cases.")
    except Exception as e:
        click.echo(f"❌ Submission is invalid: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()