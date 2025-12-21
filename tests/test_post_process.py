import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

from src.record.analysis import BenchmarkAnalyzer
from src.record.rankboard import RankBoard
from src.record.visualization import main_visualize

def verify_logic(result_file_path):
    result_path = Path(result_file_path)
    if not result_path.exists():
        logger.error(f"Result file not found: {result_path}")
        return

    output_dir = result_path.parent
    logger.info(f"📍 Testing with result file: {result_path}")
    logger.info(f"📂 Outputs will be saved to: {output_dir}")

    # ================= 1. 测试 Analysis (统计分析) =================
    logger.info("\n[1/4] Testing Analysis Module...")
    try:
        # 假设 Analyzer 接受文件路径列表
        analyzer = BenchmarkAnalyzer([str(result_path)])
        stats_df = analyzer.get_metric_stats()
        
        if stats_df.empty:
            logger.warning("⚠️ Analysis returned empty DataFrame. (Might differ based on json content)")
        else:
            print(stats_df.to_markdown(index=False) if hasattr(stats_df, 'to_markdown') else stats_df)
            
        # 保存分析结果
        stats_path = output_dir / "analysis_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"✅ Analysis passed. Stats saved to {stats_path}")
    except Exception as e:
        logger.error(f"❌ Analysis Module Failed: {e}")
        import traceback; traceback.print_exc()

    # ================= 2. 测试 RankBoard (排行榜生成) =================
    logger.info("\n[2/4] Testing RankBoard Module...")
    try:
        ranker = RankBoard([str(result_path)])
        rank_csv_path = output_dir / "leaderboard.csv"
        
        # 生成排名
        rank_df = ranker.generate_rank(output_csv=str(rank_csv_path))
        
        if rank_df is not None and not rank_df.empty:
            print(rank_df.to_markdown(index=False) if hasattr(rank_df, 'to_markdown') else rank_df)
            logger.info(f"✅ RankBoard passed. CSV saved to {rank_csv_path}")
        else:
            logger.warning("⚠️ RankBoard produced empty result.")
            
    except Exception as e:
        logger.error(f"❌ RankBoard Module Failed: {e}")
        import traceback; traceback.print_exc()

    # ================= 3. 测试 Visualization (雷达图) =================
    logger.info("\n[3/4] Testing Visualization Module...")
    try:
        vis_output_path = output_dir / "radar_chart.png"
        
        # 注意: 你的结果只有 FVD，单一指标画雷达图可能会报警或很难看，但不能报错
        main_visualize([str(result_path)], output_img=str(vis_output_path))
        
        if vis_output_path.exists():
            logger.info(f"✅ Visualization passed. Chart saved to {vis_output_path}")
        else:
            logger.warning("⚠️ Visualization function ran but no file was created.")
            
    except Exception as e:
        logger.warning(f"⚠️ Visualization Warning (Expected if too few metrics): {e}")

    # ================= 4. 模拟 HuggingFace App 数据加载 =================
    logger.info("\n[4/4] Verifying HF App Compatibility...")
    try:
        # 模拟 src/app/HuggingFace/leaderboard.py 的读取逻辑
        # 通常 HF App 读取的是 RankBoard 生成的 leaderboard.csv
        target_csv = output_dir / "leaderboard.csv"
        
        if not target_csv.exists():
            logger.error("❌ Cannot verify HF logic: leaderboard.csv missing.")
        else:
            df = pd.read_csv(target_csv)
            # 检查关键列是否存在
            required_cols = ['Model', 'Total_Score'] # 根据你的 RankBoard 实现调整
            if 'Rank' in df.columns: required_cols.append('Rank')
            
            missing_cols = [c for c in required_cols if c not in df.columns]
            
            if not missing_cols:
                logger.info(f"✅ HF App Data Check Passed. Columns found: {list(df.columns)}")
                logger.info("   -> This CSV is ready to be pushed to HuggingFace Dataset.")
            else:
                logger.warning(f"⚠️ HF App Data Check Warning: Missing columns {missing_cols}")

    except Exception as e:
        logger.error(f"❌ HF Logic Verification Failed: {e}")

if __name__ == "__main__":
    # 指向你刚刚生成的结果文件
    # 请根据实际情况修改路径
    RESULT_FILE = "output/2025-12-21_17-14-36_results.json" 
    
    # 自动搜索 output 下最新的 json (方便)
    if not os.path.exists(RESULT_FILE):
        output_dir = Path("output")
        if output_dir.exists():
            jsons = list(output_dir.glob("*_results.json"))
            if jsons:
                # 按修改时间排序取最新的
                RESULT_FILE = str(sorted(jsons, key=os.path.getmtime)[-1])
    
    verify_logic(RESULT_FILE)