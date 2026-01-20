# visualizer.py
import streamlit as st
import pandas as pd
import folium
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium
from config import *
from utils import haversine_vectorized  # 需要引入距离计算用于诊断

# 页面设置
st.set_page_config(layout="wide", page_title="智能点位规划系统")
st.title("🗺️ 智能点位规划可视化系统")


@st.cache_data
def load_data():
    try:
        details = pd.read_csv('output_details.csv')
        centers = pd.read_csv('output_centers.csv')
        return details, centers
    except FileNotFoundError:
        return None, None


def create_map(_city_shops, _city_centers):
    # 1. 地图初始化
    if len(_city_shops) > 0:
        map_center = [_city_shops[COL_LAT].mean(), _city_shops[COL_LNG].mean()]
    else:
        map_center = [39.9, 116.4]

    folium_map = folium.Map(
        location=map_center,
        zoom_start=11,
        tiles="CartoDB positron",
        prefer_canvas=True
    )

    # --- 创建图层组 (用于右上角开关) ---
    fg_high = folium.FeatureGroup(name="🔴 高负载站点 (≥90%)", show=True)
    fg_mid = folium.FeatureGroup(name="🟠 中负载站点 (50-90%)", show=True)
    fg_low = folium.FeatureGroup(name="🔵 低负载站点 (<50%)", show=True)

    fg_circles_high = folium.FeatureGroup(name="⭕ 高负载覆盖范围", show=True)
    fg_circles_mid = folium.FeatureGroup(name="⭕ 中负载覆盖范围", show=True)
    fg_circles_low = folium.FeatureGroup(name="⭕ 低负载覆盖范围", show=True)

    # --- 诊断图层: 解释为什么不合并 (默认关闭) ---
    fg_diagnosis = folium.FeatureGroup(name="🔍 合并潜力诊断", show=False)

    # 2. 绘制站点和圆圈
    for row in _city_centers.itertuples():
        load = getattr(row, 'load', 0)
        capacity_rate = getattr(row, 'capacity_rate', 0)
        center_id = getattr(row, 'center_id', 'Unknown')

        tooltip_html = f"""
        <div style="font-family: sans-serif;">
            <b>ID:</b> {center_id}<br>
            <b>负载:</b> {load}/{MAX_CAPACITY} ({capacity_rate:.0%})<br>
            <b>半径:</b> {row.radius:.2f}km
        </div>
        """

        # 分类逻辑
        if capacity_rate >= 0.9:
            target_fg = fg_high
            target_circle_fg = fg_circles_high
            color = 'darkred'
        elif capacity_rate >= 0.5:
            target_fg = fg_mid
            target_circle_fg = fg_circles_mid
            color = 'orange'
        else:
            target_fg = fg_low
            target_circle_fg = fg_circles_low
            color = 'blue'

        # 添加中心点 (十字准星)
        folium.Marker(
            location=[row.lat, row.lng],
            icon=folium.Icon(color=color, icon='crosshairs', prefix='fa'),
            tooltip=tooltip_html,
            z_index_offset=1000
        ).add_to(target_fg)

        # 添加覆盖圆
        folium.Circle(
            location=[row.lat, row.lng],
            radius=row.radius * 1000,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.05,
            weight=1,
            popup=f"半径: {row.radius:.2f}km"
        ).add_to(target_circle_fg)

    # 3. 绘制诊断线 (解释低负载原因)
    # 获取当前城市的半径限制
    if 'city_tier' in _city_centers.columns and len(_city_centers) > 0:
        tier = _city_centers.iloc[0]['city_tier']
        current_radius_limit = TIER_RADIUS_LIMIT.get(tier, DEFAULT_RADIUS_LIMIT)
    else:
        current_radius_limit = 3.0  # 默认值

    # 提取低负载站点
    low_load_centers = _city_centers[_city_centers['capacity_rate'] < 0.5].to_dict('records')

    for i in range(len(low_load_centers)):
        center1 = low_load_centers[i]
        for j in range(i + 1, len(low_load_centers)):
            center2 = low_load_centers[j]

            # 距离计算
            dist = haversine_vectorized(center1['lng'], center1['lat'], center2['lng'], center2['lat'])
            if dist > 10:
                continue  # 太远的忽略

            # 模拟合并后的半径 (严格包络)
            # 粗略估算重心在中间
            est_radius = (center1['radius'] + center2['radius'] + dist) / 2
            merged_load = center1['load'] + center2['load']

            # 如果负载允许合并，但半径超标 -> 画红线
            if merged_load <= MAX_CAPACITY and est_radius > current_radius_limit:
                folium.PolyLine(
                    locations=[[center1['lat'], center1['lng']], [center2['lat'], center2['lng']]],
                    color='red',
                    weight=2,
                    dash_array='5, 5',
                    opacity=0.6,
                    tooltip=f"无法合并: 负载{merged_load}🆗, 但需半径{est_radius:.1f} > 限额{current_radius_limit}"
                ).add_to(fg_diagnosis)

    # 4. 将图层添加到地图
    fg_circles_high.add_to(folium_map)
    fg_circles_mid.add_to(folium_map)
    fg_circles_low.add_to(folium_map)
    fg_high.add_to(folium_map)
    fg_mid.add_to(folium_map)
    fg_low.add_to(folium_map)
    fg_diagnosis.add_to(folium_map)  # 诊断层

    # 5. 店铺聚合层
    shop_coords = _city_shops[[COL_LAT, COL_LNG]].values.tolist()
    if shop_coords:
        FastMarkerCluster(
            shop_coords,
            name="🏪 所有店铺分布",
            overlay=True,
            control=True
        ).add_to(folium_map)

    # 添加图层控制器 (默认展开)
    folium.LayerControl(collapsed=False).add_to(folium_map)

    return folium_map


# --- 主逻辑 ---
details_df, centers_df = load_data()

if details_df is None:
    st.error("未找到数据文件。请先运行 main.py 生成结果。")
else:
    # 侧边栏
    st.sidebar.header("基础设置")
    city_list = list(details_df[COL_CITY].unique())
    selected_city = st.sidebar.selectbox("选择城市", city_list)

    # 数据切片
    city_shops = details_df[details_df[COL_CITY] == selected_city]
    city_centers = centers_df[centers_df['city'] == selected_city]

    # --- 指标计算 ---
    st.subheader(f"{selected_city} 规划概览")

    # 1. 店铺覆盖率
    if 'is_covered' in city_shops.columns:
        coverage_rate = city_shops['is_covered'].mean()
    else:
        coverage_rate = 1.0

        # 2. 平均负载率
    if not city_centers.empty:
        avg_load_rate = city_centers['capacity_rate'].mean()
    else:
        avg_load_rate = 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("店铺总数", len(city_shops))
    c2.metric("规划站点数", len(city_centers))
    c3.metric("店铺覆盖率", f"{coverage_rate:.1%}")
    c4.metric("平均负载率", f"{avg_load_rate:.1%}")

    st.caption("💡 提示：点击地图右上角的图层图标 🗺️，可以勾选显示/隐藏不同负载的站点。勾选'诊断'图层可查看无法合并的原因。")

    # 渲染地图
    with st.spinner("正在渲染地图..."):
        m = create_map(city_shops, city_centers)
        st_folium(m, width=None, height=700, returned_objects=[])
