import streamlit as st
import pandas as pd
import folium
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium
from config import *

# 页面设置
st.set_page_config(layout="wide", page_title="智能点位规划系统")
st.title("🗺️ 智能点位规划可视化系统")

# --- 1. Session State 初始化 ---
if 'selected_center_id' not in st.session_state:
    st.session_state.selected_center_id = None


@st.cache_data
def load_data():
    try:
        details = pd.read_csv('output_details.csv')
        centers = pd.read_csv('output_centers.csv')
        return details, centers
    except FileNotFoundError:
        return None, None


def create_map(city_name, _city_shops, _city_centers, highlight_id=None):
    # 1. 地图初始化
    if len(_city_shops) > 0:
        map_center = [_city_shops[COL_LAT].mean(), _city_shops[COL_LNG].mean()]
    else:
        map_center = [39.9, 116.4]

    m = folium.Map(
        location=map_center,
        zoom_start=11,
        tiles="CartoDB positron",
        prefer_canvas=True
    )

    # --- 图层组 ---
    fg_high = folium.FeatureGroup(name="🔴 高负载站点 (≥90%)", show=True)
    fg_mid = folium.FeatureGroup(name="🟠 中负载站点 (50-90%)", show=True)
    fg_low = folium.FeatureGroup(name="🔵 低负载站点 (<50%)", show=True)

    fg_circles_high = folium.FeatureGroup(name="⭕ 高负载覆盖范围", show=True)
    fg_circles_mid = folium.FeatureGroup(name="⭕ 中负载覆盖范围", show=True)
    fg_circles_low = folium.FeatureGroup(name="⭕ 低负载覆盖范围", show=True)

    # 2. 绘制站点
    for row in _city_centers.itertuples():
        load = getattr(row, 'load', 0)
        capacity_rate = getattr(row, 'capacity_rate', 0)
        center_id = getattr(row, 'center_id', 'Unknown')

        # Tooltip 内容
        tooltip_html = f"""
        ID: {center_id}\n
        负载: {load}/{MAX_CAPACITY}\n
        半径: {row.radius:.2f}km
        """

        # 颜色逻辑
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

        # 高亮逻辑
        icon_type = 'crosshairs'
        if highlight_id and center_id == highlight_id:
            color = 'green'  # 选中变绿
            icon_type = 'star'

        # 添加点
        folium.Marker(
            location=[row.lat, row.lng],
            icon=folium.Icon(color=color, icon=icon_type, prefix='fa'),
            tooltip=tooltip_html,
            z_index_offset=1000
        ).add_to(target_fg)

        # 添加圆
        if highlight_id is None or center_id == highlight_id:
            folium.Circle(
                location=[row.lat, row.lng],
                radius=row.radius * 1000,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.1 if highlight_id is None else 0.3,
                weight=1,
                popup=f"半径: {row.radius:.2f}km"
            ).add_to(target_circle_fg)

    # 3. 添加图层
    fg_circles_high.add_to(m)
    fg_circles_mid.add_to(m)
    fg_circles_low.add_to(m)
    fg_high.add_to(m)
    fg_mid.add_to(m)
    fg_low.add_to(m)

    # 4. 店铺绘制逻辑
    if highlight_id:
        # A. 选中模式：只画覆盖的店铺
        target_shops = _city_shops[_city_shops['center_id'] == highlight_id]
        shop_coords = target_shops[[COL_LAT, COL_LNG]].values.tolist()

        if shop_coords:
            for lat, lng in shop_coords:
                folium.CircleMarker(
                    location=[lat, lng],
                    radius=3,
                    color='green',
                    fill=True,
                    fill_color='green',
                    fill_opacity=1.0,
                    popup="覆盖店铺"
                ).add_to(m)
    else:
        # B. 全局模式：聚合显示
        shop_coords = _city_shops[[COL_LAT, COL_LNG]].values.tolist()
        if shop_coords:
            FastMarkerCluster(
                shop_coords,
                name="🏪 所有店铺分布",
                overlay=True,
                control=True
            ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# --- 主逻辑 ---
details_df, centers_df = load_data()

if details_df is None:
    st.error("未找到数据文件。请先运行 main.py 生成结果。")
else:
    # 侧边栏
    st.sidebar.header("基础设置")
    city_list = list(details_df[COL_CITY].unique())

    selected_city = st.sidebar.selectbox("选择城市", city_list)
    # 切换城市重置状态
    if 'last_city' not in st.session_state or st.session_state.last_city != selected_city:
        st.session_state.selected_center_id = None
        st.session_state.last_city = selected_city

    # 数据切片
    city_shops = details_df[details_df[COL_CITY] == selected_city]
    city_centers = centers_df[centers_df['city'] == selected_city]

    # --- 指标区域 ---
    st.subheader(f"{selected_city} 规划概览")

    if st.session_state.selected_center_id:
        # 选中状态
        current_center = city_centers[city_centers['center_id'] == st.session_state.selected_center_id]
        if not current_center.empty:
            row = current_center.iloc[0]
            st.info(f"🔍 当前选中: **{row['center_id']}** | 负载: {row['load']} | 半径: {row['radius']:.2f}km")

        if st.button("🔙 返回全局视图", type="primary"):
            st.session_state.selected_center_id = None
            st.rerun()
    else:
        # 全局状态
        if 'is_covered' in city_shops.columns:
            coverage_rate = city_shops['is_covered'].mean()
        else:
            coverage_rate = 1.0

        if not city_centers.empty:
            avg_load_rate = city_centers['capacity_rate'].mean()
        else:
            avg_load_rate = 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("店铺总数", len(city_shops))
        c2.metric("规划站点数", len(city_centers))
        c3.metric("店铺覆盖率", f"{coverage_rate:.1%}")
        c4.metric("平均负载率", f"{avg_load_rate:.1%}")

    st.caption("💡 提示：点击地图上的**站点图标**（十字准星），即可查看该站点覆盖的店铺细节。")

    # 渲染地图
    with st.spinner("正在渲染地图..."):
        m = create_map(selected_city, city_shops, city_centers, st.session_state.selected_center_id)

        # --- 关键修改：只获取 last_object_clicked ---
        # 我们不需要 tooltip 文本了，直接要坐标
        map_data = st_folium(m, width=None, height=700, returned_objects=["last_object_clicked"])

    # --- 交互逻辑 (坐标匹配法) ---
    if map_data and map_data.get("last_object_clicked"):
        clicked_obj = map_data["last_object_clicked"]

        if clicked_obj:
            lat = clicked_obj['lat']
            lng = clicked_obj['lng']

            # 在 city_centers 里找坐标匹配的点
            # 为了防止浮点数精度问题，使用一个小范围 (epsilon)
            epsilon = 0.0001

            match = city_centers[
                (city_centers['lat'] > lat - epsilon) &
                (city_centers['lat'] < lat + epsilon) &
                (city_centers['lng'] > lng - epsilon) &
                (city_centers['lng'] < lng + epsilon)
                ]

            if not match.empty:
                target_id = match.iloc[0]['center_id']

                # 如果点击了新的点，更新并刷新
                if target_id != st.session_state.selected_center_id:
                    st.session_state.selected_center_id = target_id
                    st.rerun()
