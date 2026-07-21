# Smart Site Planner

[English](README.md) | Simplified Chinese

Smart Site Planner 是一个 Python 与 Streamlit 项目，用 POI 坐标进行配送站点覆盖规划、输出站点分配结果，并在交互式地图中查看结果。

本仓库反映协作式课程/项目工作。贡献时请保留已有团队和课程署名，不要将项目表述为单人作品。

## 快速开始

```bash
git clone https://github.com/JojoZhu9/smart-site-planner.git
cd smart-site-planner
pip install -r requirements.txt
```

准备 `data/data.txt` 后运行：

```bash
python main.py
streamlit run src/visualizer.py
```

规划程序生成 `data/output_centers.csv` 与 `data/output_details.csv`；生成完成后再启动看板。

## 数据格式与配置

默认输入路径是 `data/data.txt`，定义在 `src/config.py`。程序先按制表符读取；若没有 `tbsg_latitude` 列，则按逗号分隔重试。请勿提交运营 POI 或位置数据。

| 列名 | 含义 | 是否必需 |
| --- | --- | --- |
| `tbsg_longitude` | 门店经度 | 是 |
| `tbsg_latitude` | 门店纬度 | 是 |
| `second_district_name` | 用于拆分规划任务的城市或二级行政区 | 是 |

无效经纬度会被丢弃。当前配置中 `COL_SALES` 与 `COL_CITY_TIER` 为空，因此默认不启用销售额和城市等级逻辑。

在 [src/config.py](src/config.py) 中调整路径、列名和规划参数。默认值为：

| 设置 | 默认值 | 作用 |
| --- | --- | --- |
| `DATA_PATH` | `./data/data.txt` | 输入数据 |
| `MAX_CAPACITY` | `120` | 单个最终站点的最大门店数 |
| `DEFAULT_RADIUS_LIMIT` | `3.0` km | 默认覆盖半径 |
| `MERGE_DISTANCE_THRESHOLD` | `1.0` km | 候选站点融合阈值 |

## 算法与输出

`main.py` 会按非空行政区处理数据：先分别生成容量为 50 和 120 的候选站点，再以 `MERGE_DISTANCE_THRESHOLD` 融合候选点，最后以容量 120 进行优化。求解器使用空间邻域查询、贪心聚类、最小包围圆以及吸收、邻近合并、小站点合并和未覆盖门店分配等后处理。

| 文件 | 内容 |
| --- | --- |
| `data/output_centers.csv` | 最终站点标识、行政区和行政区等级、坐标、半径、负载、容量率、站点销售额和来源类型 |
| `data/output_details.csv` | 可视化所需的逐门店规划字段，以及分配的站点标识、覆盖状态和到站点距离 |
| `logs/solver_run.log` | `main.py` 生成的运行日志 |

这些文件均为本地生成内容，已被 Git 忽略。`src/visualizer.py` 会读取两个 CSV 输出，按行政区筛选并在 Streamlit/Folium 中显示站点、覆盖圆与门店点。

## 截图

### 主界面
![主界面](images/main_interface.png)

### 站点覆盖
![站点覆盖](images/site_coverage.png)

## 验证与限制

可用的语法检查命令是：

```bash
python -m compileall main.py src test_main.py
```

`test_main.py` 是用于比较融合阈值的实验运行器，不是 pytest 测试模块。仓库当前没有可被 pytest 收集的测试用例，因此 `python -m pytest test_main.py -q` 会因收集到零个测试而退出。这是仓库限制，不代表已有测试覆盖率。仓库也没有附带示例数据；只应使用已获授权、且符合上述格式的数据集。

## 贡献与安全

提交前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。敏感数据、凭据或依赖问题请按 [SECURITY.md](SECURITY.md) 私下报告，不要公开提交。

## 项目结构

```text
smart-site-planner/
├── images/                 # 已跟踪的看板截图
├── src/                    # 配置、合并、求解、工具与可视化
├── main.py                 # 批量规划入口
├── test_main.py            # 阈值实验运行器
└── requirements.txt        # Python 依赖
```
