import pandas as pd
import os
import time
import logging
import io
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.solver import CoverageSolver
from src.merger import CandidateMerger
from src.config import *

# --- 实验配置 ---
EXPERIMENT_THRESHOLDS = [0.5, 1.0, 1.5, 2.0]
LIMIT_CITIES = 20
LOG_DIR = "./logs"


def process_single_city_pipeline(city_name, city_df, max_capacity, merge_threshold):
    log_capture = io.StringIO()
    centers = []
    details = pd.DataFrame()
    status = "Fail"
    n_centers = 0
    c120_n, c50_n, merged_n = 0, 0, 0

    # 捕获标准输出，但保留错误堆栈
    with contextlib.redirect_stdout(log_capture):
        try:
            # 1. C50
            s50 = CoverageSolver(city_df, max_capacity=50)
            s50.solve(use_existing_init=False)
            c50 = s50.final_centers
            c50_n = len(c50)

            # 2. C120
            s120 = CoverageSolver(city_df, max_capacity=120)
            s120.solve(use_existing_init=False)
            c120 = s120.final_centers
            c120_n = len(c120)

            # 3. 融合
            merged = CandidateMerger.merge_and_prune(c120, c50, distance_threshold_km=merge_threshold)
            merged_n = len(merged)

            # 4. 最终优化
            final_solver = CoverageSolver(city_df, max_capacity=max_capacity)
            final_solver.load_external_candidates(merged)

            # 【修复】兼容 2 个或 3 个返回值
            solve_res = final_solver.solve(use_existing_init=True)
            if len(solve_res) == 3:
                centers, details, _ = solve_res
            else:
                centers, details = solve_res

            n_centers = len(centers)
            status = "Success"

        except Exception as e:
            status = "Error"
            import traceback
            # 返回错误信息以便调试
            return city_name, "Error", 0, 0, 0, 0, traceback.format_exc()

    return city_name, status, n_centers, c120_n, c50_n, merged_n, None


def run_experiment_round(threshold, city_tasks, max_workers):
    print(f"\n🧪 [实验开始] 测试阈值: {threshold} km ...")

    total_c120 = 0
    total_c50 = 0
    total_merged = 0
    total_final = 0
    success_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_city_pipeline, city, data, 120, threshold): city
            for city, data in city_tasks
        }

        pbar = tqdm(as_completed(futures), total=len(city_tasks), unit="city", ncols=80)

        for future in pbar:
            try:
                res = future.result()
                if len(res) == 7:
                    city_name, status, n_centers, c120_n, c50_n, merged_n, error_msg = res
                else:
                    city_name, status, n_centers, c120_n, c50_n, merged_n = res
                    error_msg = None

                if status == "Success":
                    total_c120 += c120_n
                    total_c50 += c50_n
                    total_merged += merged_n
                    total_final += n_centers
                    success_count += 1
                else:
                    if error_msg:
                        print(f"\n❌ {city_name} 报错:\n{error_msg}")
                        break
            except Exception as e:
                print(f"Future Error: {e}")

    return {
        "Threshold": threshold,
        "Final_Sites": total_final,
        "Merged_Seeds": total_merged,
        "Reduction_Total": (total_c120 + total_c50) - total_final,
        "Reduction_Merge": (total_c120 + total_c50) - total_merged
    }


def main():
    if not os.path.exists(DATA_PATH): return
    try:
        df = pd.read_csv(DATA_PATH, sep='\t')
        if 'tbsg_latitude' not in df.columns: df = pd.read_csv(DATA_PATH, sep=',')
    except:
        return

    unique_cities = df[COL_CITY].unique()
    city_tasks = []
    for city in unique_cities:
        if pd.isna(city) or str(city).strip() == "": continue
        sub_df = df[df[COL_CITY] == city].copy()
        if len(sub_df) < 1: continue
        city_tasks.append((city, sub_df))

    if LIMIT_CITIES: city_tasks = city_tasks[:LIMIT_CITIES]
    max_workers = 4

    results = []
    for t in EXPERIMENT_THRESHOLDS:
        stats = run_experiment_round(t, city_tasks, max_workers)
        results.append(stats)

    print("\n" + "=" * 60)
    print("📊 消融实验最终报告")
    print("=" * 60)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()