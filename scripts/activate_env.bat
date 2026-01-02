@echo off
REM activate_env.bat - 快速激活 bench 环境

echo 激活 bench 环境...
call conda activate bench

if "%CONDA_DEFAULT_ENV%"=="bench" (
    echo 环境激活成功！当前环境: %CONDA_DEFAULT_ENV%
    echo.
    echo 可用命令:
    echo   python main.py --help          查看主程序帮助
    echo   python -m pytest tests/       运行测试
    echo   conda deactivate              退出环境
    echo.
) else (
    echo 环境激活失败！请检查 bench 环境是否已正确安装。
    echo 如果未安装，请运行: scripts\build_env_windows.ps1
    echo 或者运行: scripts\build_env.bat
)

cmd /k
