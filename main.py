import warnings
import pandas as pd
from src.solver import CoverageSolver
from src.config import DATA_PATH, OUTPUT_CENTERS, OUTPUT_DETAILS

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    # 读取数据
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"错误：找不到文件 {DATA_PATH}")
        return

    print("数据读取成功，开始计算点位覆盖...")

    solver = CoverageSolver(df)
    centers_df, result_df = solver.solve()

    # 保存点位结果和详细归属结果
    centers_df.to_csv(OUTPUT_CENTERS, index=False)
    result_df.to_csv(OUTPUT_DETAILS, index=False)

    print(f"计算完成！")
    print(f"共生成点位: {len(centers_df)} 个")
    print(f"覆盖率: {result_df['is_covered'].mean():.2%}")
    print("✅ 完成！结果已保存至 data/ 目录")


if __name__ == "__main__":
    main()
