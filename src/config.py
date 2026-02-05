# --- 文件路径 ---
DATA_PATH = './data/data.txt'
OUTPUT_CENTERS = './data/output_centers.csv'
OUTPUT_DETAILS = './data/output_details.csv'

# --- 列名映射 ---
COL_LNG = 'tbsg_longitude'           # 经度
COL_LAT = 'tbsg_latitude'            # 纬度
COL_CITY = 'second_district_name'    # 城市/二级行政区 (例如: 北京市, 朝阳区等)
COL_SALES = ''
COL_CITY_TIER = ''

# --- 算法参数 ---
MAX_CAPACITY = 120                    # 单个站点最大店铺数
DEFAULT_RADIUS_LIMIT = 3.0            # 默认半径限制 (km)
MERGE_DISTANCE_THRESHOLD = 1.0        # 融合去重距离阈值 (km)

# 城市等级对应的半径限制 (km)
TIER_RADIUS_LIMIT = {
    '超大城市': 4.0,
    '特大城市': 3.5,
    '大城市': 3.0,
    '中等城市': 2.5,
    '小城市': 2.0,
    '未知': 3.0  # 默认兜底
}
