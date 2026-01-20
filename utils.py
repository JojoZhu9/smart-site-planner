import numpy as np

R_EARTH = 6371.0  # 地球半径 km


def haversine_vectorized(lon1, lat1, lon2, lat2):
    """
    向量化计算两点间距离 (km)
    输入可以是浮点数或numpy数组
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    don = lon2 - lon1
    dat = lat2 - lat1

    a = np.sin(dat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(don / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return c * R_EARTH
