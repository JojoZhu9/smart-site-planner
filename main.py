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

# --- 1. 仅定义常量 (不要在这里做文件操作！) ---
LOG_DIR = "./logs"
LOG_FILE = os.path.join(LOG_DIR, "solver_run.log")


def process_single_city_pipeline(city_name, city_df, max_capacity):
    """
    单个城市的完整处理流水线。
    注意：子进程中不要配置 logging 到文件，只捕获 stdout。
    """
    # 创建字符串缓冲区捕获输出
    log_capture = io.StringIO()

    centers = []
    details = pd.DataFrame()
    status = "Fail"
    n_centers = 0

    # 【新增】统计变量
    count_120 = 0
    count_50 = 0
    count_merged = 0

    # 重定向 stdout 到缓冲区
    with contextlib.redirect_stdout(log_capture):
        try:
            print(f"[{time.strftime('%H:%M:%S')}] === 开始处理: {city_name} (Rows: {len(city_df)}) ===")

            # 1. 阶段一：Capacity=50
            solver_50 = CoverageSolver(city_df, max_capacity=50)
            solver_50.solve(use_existing_init=False)
            c50 = solver_50.final_centers
            count_50 = len(c50)  # 记录数量

            # 2. 阶段二：Capacity=120
            solver_120 = CoverageSolver(city_df, max_capacity=120)
            solver_120.solve(use_existing_init=False)
            c120 = solver_120.final_centers
            count_120 = len(c120)  # 记录数量

            # 3. 阶段三：融合
            merged = CandidateMerger.merge_and_prune(c120, c50, distance_threshold_km=MERGE_DISTANCE_THRESHOLD)
            count_merged = len(merged)  # 记录数量

            # 4. 阶段四：最终优化
            final_solver = CoverageSolver(city_df, max_capacity=max_capacity)
            final_solver.load_external_candidates(merged)
            centers, details = final_solver.solve(use_existing_init=True)

            n_centers = len(centers)
            status = "Success"
            print(f"[{time.strftime('%H:%M:%S')}] === 处理完成: {city_name} | 最终站点: {n_centers} ===")
            print("-" * 30)

        except Exception as e:
            status = "Error"
            print(f"❌ 异常: {str(e)}")
            import traceback
            traceback.print_exc()

    # 返回所有结果，包括新增的统计数据
    return city_name, centers, details, status, n_centers, count_120, count_50, count_merged, log_capture.getvalue()


def main():
    # 文件操作和日志配置移入 main 函数
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
        except PermissionError:
            print("⚠️ 警告: 无法删除旧日志文件(可能被占用)，将追加写入。")

    # 配置主进程日志 (只有主进程写文件)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(message)s",
        encoding='utf-8'
    )

    start_time = time.time()
    print(f"🚀 启动智能选址系统...")
    print(f"📂 日志存放于: {LOG_FILE}")

    # 1. 读取全量数据
    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到文件: {DATA_PATH}")
        return

    print(f"⏳ 正在读取数据...")
    try:
        df = pd.read_csv(DATA_PATH, sep='\t')
        if 'tbsg_latitude' not in df.columns:
            df = pd.read_csv(DATA_PATH, sep=',')
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 2. 按城市拆分数据
    unique_cities = df[COL_CITY].unique()
    city_tasks = []
    for city in unique_cities:
        if pd.isna(city) or str(city).strip() == "": continue
        sub_df = df[df[COL_CITY] == city].copy()
        if len(sub_df) < 1: continue
        city_tasks.append((city, sub_df))

    total_cities = len(city_tasks)
    print(f"✅ 数据加载完毕，共 {total_cities} 个城市任务。")

    # 3. 并行执行
    # Windows下建议不要占满所有CPU，留1-2个核给系统
    max_workers = max(1, min(os.cpu_count() - 2, 20))
    print(f"🔥 启动进程池 (Workers={max_workers})...")

    all_centers = []
    all_details = []
    success_count = 0
    error_count = 0

    # 【新增】全局统计变量
    total_c120 = 0
    total_c50 = 0
    total_merged = 0
    total_final = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_city_pipeline, city, data, 120): city
            for city, data in city_tasks
        }

        # 进度条
        pbar = tqdm(as_completed(futures), total=total_cities, unit="city", ncols=100)

        for future in pbar:
            try:
                # 获取子进程返回的结果 (解包新增的变量)
                city_name, centers, details, status, n_centers, c120_n, c50_n, merged_n, log_str = future.result()

                # 1. 将子进程的日志写入主日志文件
                logging.info(log_str)

                # 2. 处理业务数据
                if status == "Success":
                    all_centers.append(centers)
                    all_details.append(details)
                    success_count += 1

                    # 累加统计
                    total_c120 += c120_n
                    total_c50 += c50_n
                    total_merged += merged_n
                    total_final += n_centers

                    pbar.set_postfix_str(f"Last: {city_name} ({n_centers}站) | Err: {error_count}")
                else:
                    error_count += 1
                    pbar.set_postfix_str(f"Last: {city_name} [ERR] | Err: {error_count}")
            except Exception as e:
                error_count += 1
                print(f"\n❌ 主进程处理结果时异常: {e}")

    # 4. 合并结果
    print("\n💾 正在保存结果...")
    if all_centers:
        final_centers_df = pd.concat(all_centers, ignore_index=True)
        final_details_df = pd.concat(all_details, ignore_index=True)

        # 生成全局唯一ID
        final_centers_df['center_id'] = [f"C_{i + 1:06d}" for i in range(len(final_centers_df))]

        final_centers_df.to_csv(OUTPUT_CENTERS, index=False, encoding='utf-8-sig')
        final_details_df.to_csv(OUTPUT_DETAILS, index=False, encoding='utf-8-sig')

        duration = time.time() - start_time

        # 【新增】计算统计指标
        total_initial = total_c120 + total_c50
        reduced_by_merge = total_initial - total_merged  # 融合阶段剔除
        reduced_by_opt = total_merged - total_final  # 最终优化剔除
        total_reduced = total_initial - total_final  # 总剔除

        print("-" * 50)
        print(f"✅ 全部完成！")
        print(f"⏱️ 总耗时: {duration:.1f}s")
        print(f"🏙️ 成功: {success_count} | 失败: {error_count}")
        print(f"📍 总站点数: {len(final_centers_df)}")
        print(f"📂 结果保存: data/")
        print(f"📝 运行日志: {LOG_FILE}")

        print("-" * 50)
        print(f"📊 算法优化统计报表:")
        print(f"   1. 初始生成池:")
        print(f"      - C120 (大站): {total_c120}")
        print(f"      - C50  (小站): {total_c50}")
        print(f"      - 合计初始点 : {total_initial}")
        print(f"   2. 融合与优化:")
        print(f"      - 融合后候选点: {total_merged} (📉 融合剔除: {reduced_by_merge})")
        print(f"      - 最终优选站点: {total_final}  (📉 优化剔除: {reduced_by_opt})")
        print(f"   3. 总体效果:")
        print(f"      - 总计减少冗余点位: {total_reduced}")
        print("-" * 50)
    else:
        print("❌ 未生成任何结果，请检查 logs/solver_run.log")


if __name__ == "__main__":
    main()
