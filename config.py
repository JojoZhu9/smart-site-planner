# 数据读取地址
DATA_PATH = 'data.csv'

# 列名映射
COL_SHOP_ID = '商家ID'
COL_CITY = '城市名称'
COL_LNG = '经度'
COL_LAT = '纬度'
COL_SALES = '30天销量'
COL_CITY_TIER = '城市等级'

# 核心约束
MAX_CAPACITY = 120  # 绝对硬指标：单点最大负载

# 城市等级 -> 半径上限 (只看 Max，因为现在是切分逻辑)
TIER_RADIUS_LIMIT = {
    '超大城市': 5.0,
    '大城市':   4.5,
    '中等城市': 4.0,
    '小城市':  3.5,
    '微型城市': 3.0,
}
# 默认上限
DEFAULT_RADIUS_LIMIT = 4.0
