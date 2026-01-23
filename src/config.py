# --- 文件路径 ---
DATA_PATH = './data/data_test.csv'  # 修改文件名
OUTPUT_CENTERS = './data/output_centers.csv'
OUTPUT_DETAILS = './data/output_details.csv'

# --- 列名映射 (根据新表头修改) ---
COL_LNG = 'longitude'           # 经度
COL_LAT = 'latitude'            # 纬度
COL_CITY = 'poi_second_district_name'  # 城市/二级行政区 (例如: 北京市, 朝阳区等)
# 注意: 你的数据里好像没有明确的 "销量" 列，如果有，请填写真实的列名
# 如果没有销量数据，代码会自动处理(视为全1或忽略销量加权)
COL_SALES = ''
# --- 城市等级配置 (可选) ---
# 如果你的数据里没有 city_tier 列，代码会默认使用 '未知' 等级
# 你可以手动指定这一列，或者在代码里写死逻辑
COL_CITY_TIER = 'city_tier'     # 数据中似乎没有这一列，保持默认即可

# --- 算法参数 ---
MAX_CAPACITY = 50              # 单个站点最大店铺数
DEFAULT_RADIUS_LIMIT = 3.0      # 默认半径限制 (km)

# 城市等级对应的半径限制 (km)
TIER_RADIUS_LIMIT = {
    '超大城市': 4.0,
    '特大城市': 3.5,
    '大城市': 3.0,
    '中等城市': 2.5,
    '小城市': 2.0,
    '未知': 3.0  # 默认兜底
}
