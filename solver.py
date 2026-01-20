import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from config import *
from utils import haversine_vectorized
import sys

# 增加递归深度限制，防止极端情况
sys.setrecursionlimit(2000)


class CoverageSolver:
    def __init__(self, df):
        self.df = df.copy()
        self.df[COL_LNG] = self.df[COL_LNG].astype(float)
        self.df[COL_LAT] = self.df[COL_LAT].astype(float)

        self.final_centers = []
        self.shop_assignments = {}

    @staticmethod
    def get_radius_limit(city_df):
        try:
            tier = city_df.iloc[0][COL_CITY_TIER]
        except (KeyError, IndexError):
            tier = '未知'
        return TIER_RADIUS_LIMIT.get(tier, DEFAULT_RADIUS_LIMIT), tier

    def save_cluster(self, df_subset, city_name, city_tier, center_lat, center_lng, radius):
        """辅助函数：保存一个簇为最终结果"""
        center_id = f"Auto_{len(self.final_centers) + 1}"

        display_radius = max(radius, 0.1)

        # 记录中心点
        self.final_centers.append({
            'center_id': center_id,
            'city': city_name,
            'city_tier': city_tier,
            'lat': center_lat,
            'lng': center_lng,
            'radius': display_radius,
            'load': len(df_subset),
            'capacity_rate': len(df_subset) / MAX_CAPACITY,
            # 虚拟站点的销量 = 辖区内店铺销量之和
            'center_sales': df_subset[COL_SALES].sum() if COL_SALES in df_subset.columns else 0
        })

        # 记录归属
        # 计算每个点到中心的距离
        if len(df_subset) > 0:
            dists = haversine_vectorized(
                center_lng, center_lat,
                df_subset[COL_LNG].values, df_subset[COL_LAT].values
            )
            for idx, dist in zip(df_subset.index, dists):
                self.shop_assignments[idx] = {
                    'center_id': center_id,
                    'distance': dist
                }

    def recursive_cluster(self, df_subset, radius_limit, city_name, city_tier, depth=0):
        """
        递归聚类 (带防死循环机制)
        depth: 当前递归深度
        """
        n_points = len(df_subset)

        # --- 0. 基础出口 ---
        if n_points == 0:
            return

        # --- 1. 计算几何参数 ---
        center_lat = df_subset[COL_LAT].mean()
        center_lng = df_subset[COL_LNG].mean()

        if n_points > 1:
            dists = haversine_vectorized(
                center_lng, center_lat,
                df_subset[COL_LNG].values, df_subset[COL_LAT].values
            )
            current_radius = dists.max()
        else:
            current_radius = 0

        # --- 2. 检查是否满足条件 ---
        is_load_ok = (n_points <= MAX_CAPACITY)
        is_radius_ok = (current_radius <= radius_limit)

        # 安全刹车：如果递归太深(超过50层)，或者只剩很少的点，强制停止
        if (is_load_ok and is_radius_ok) or n_points <= 1 or depth > 50:
            self.save_cluster(df_subset, city_name, city_tier, center_lat, center_lng, current_radius)
            return

        # --- 3. 特殊情况处理：高密度重叠点 (The Stacked Points Problem) ---
        # 如果半径非常小（说明点都在一起），但数量很多（超载）
        # 这时候 K-Means 是分不开的，必须暴力切分
        if current_radius < 0.05 and not is_load_ok:
            # print(f"  [触发硬切分] {city_name} 发现 {n_points} 个重叠点，半径 {current_radius:.4f}km")

            # 简单的逻辑：直接按列表顺序切成几块
            # 这种切分不考虑地理位置（因为地理位置几乎一样），只为了满足负载限制
            num_chunks = int(np.ceil(n_points / MAX_CAPACITY))

            # 使用 numpy array_split 进行切分
            chunks = np.array_split(df_subset, num_chunks)

            for chunk in chunks:
                # 递归调用，但深度+1
                self.recursive_cluster(chunk, radius_limit, city_name, city_tier, depth + 1)
            return

        # --- 4. 正常情况：K-Means 空间分裂 ---
        # 决定分裂数量
        k_by_load = int(np.ceil(n_points / MAX_CAPACITY))
        k = max(2, k_by_load)  # 至少切2份

        # 如果是因为半径太大而分裂，尝试切多一点，让子区域半径迅速变小
        if not is_radius_ok and k == 2:
            k = 3

        k = min(k, n_points)

        try:
            coords = df_subset[[COL_LAT, COL_LNG]].values
            # n_init=3 减少随机性，batch_size 调大提高速度
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=512, n_init=3)
            labels = kmeans.fit_predict(coords)

            df_subset = df_subset.copy()  # 避免 SettingWithCopyWarning
            df_subset['label'] = labels

            # 检查 K-Means 是否真的把点分开了
            # 如果 K-Means 分裂失败（比如所有点都被分到 label 0），会导致死循环
            unique_labels = df_subset['label'].unique()
            if len(unique_labels) < 2:
                # K-Means 失败（可能是数据分布极其特殊），转为硬切分
                # 这里的逻辑同上面的“重叠点处理”
                chunks = np.array_split(df_subset.drop(columns=['label']), k)
                for chunk in chunks:
                    self.recursive_cluster(chunk, radius_limit, city_name, city_tier, depth + 1)
                return

            # 正常递归
            for label in unique_labels:
                sub_df = df_subset[df_subset['label'] == label].drop(columns=['label'])
                self.recursive_cluster(sub_df, radius_limit, city_name, city_tier, depth + 1)

        except Exception as e:
            # 万一 K-Means 报错，兜底保存当前状态，防止程序崩溃
            print(f"K-Means Error in {city_name}: {e}")
            self.save_cluster(df_subset, city_name, city_tier, center_lat, center_lng, current_radius)

    def process_city(self, city_name):
        city_df = self.df[self.df[COL_CITY] == city_name].copy()
        radius_limit, city_tier = self.get_radius_limit(city_df)

        print(f"处理 {city_name} ({city_tier}): 约束 [负载<{MAX_CAPACITY}, 半径<{radius_limit}km]")

        # 启动递归，深度从0开始
        self.recursive_cluster(city_df, radius_limit, city_name, city_tier, depth=0)

    def post_process_absorb(self):
        print("正在处理包含关系...")
        centers = self.final_centers
        centers.sort(key=lambda x: x['radius'], reverse=True)

        active_indices = set(range(len(centers)))
        new_centers_list = []

        for i in range(len(centers)):
            if i not in active_indices:
                continue

            big = centers[i]
            merged_indices = []

            for j in range(len(centers)):
                if i == j or j not in active_indices:
                    continue
                small = centers[j]

                if big['city'] != small['city']:
                    continue

                # 宽松判断包含关系
                dist = haversine_vectorized(big['lng'], big['lat'], small['lng'], small['lat'])
                if dist + small['radius'] > big['radius'] * 1.2:
                    continue

                if big['load'] + small['load'] <= MAX_CAPACITY:
                    # --- 修正开始 ---
                    # 1. 计算新的重心 (加权平均)
                    total_load = big['load'] + small['load']
                    new_lat = (big['lat'] * big['load'] + small['lat'] * small['load']) / total_load
                    new_lng = (big['lng'] * big['load'] + small['lng'] * small['load']) / total_load

                    # 2. 计算新重心到两个旧圆心的距离
                    dist_to_big = haversine_vectorized(new_lng, new_lat, big['lng'], big['lat'])
                    dist_to_small = haversine_vectorized(new_lng, new_lat, small['lng'], small['lat'])

                    # 3. 严格计算新半径：必须能包住原来的两个圆
                    new_radius = max(dist_to_big + big['radius'], dist_to_small + small['radius'])

                    # 更新大圈数据
                    big['lat'] = new_lat
                    big['lng'] = new_lng
                    big['radius'] = new_radius  # 更新半径！
                    big['load'] = total_load
                    big['capacity_rate'] = total_load / MAX_CAPACITY
                    big['center_sales'] += small['center_sales']
                    # --- 修正结束 ---

                    merged_indices.append(j)

            for idx in merged_indices:
                active_indices.remove(idx)

            if i in active_indices:
                new_centers_list.append(big)

        self.final_centers = new_centers_list

    def post_process_merge_neighbors(self):
        print("正在优化邻居站点...")
        centers = self.final_centers
        active_indices = set(range(len(centers)))
        new_centers_list = []

        for i in range(len(centers)):
            if i not in active_indices:
                continue

            current = centers[i]
            best_merge_idx = -1
            min_dist = float('inf')

            tier = current.get('city_tier', '未知')
            radius_limit = TIER_RADIUS_LIMIT.get(tier, DEFAULT_RADIUS_LIMIT)

            for j in range(len(centers)):
                if i == j or j not in active_indices:
                    continue
                neighbor = centers[j]

                if current['city'] != neighbor['city']:
                    continue
                if current['load'] + neighbor['load'] > MAX_CAPACITY:
                    continue

                # 距离
                dist = haversine_vectorized(current['lng'], current['lat'], neighbor['lng'], neighbor['lat'])

                # --- 修正开始：预计算合并后的几何属性 ---
                # 1. 预估新重心
                temp_total_load = current['load'] + neighbor['load']
                temp_lat = (current['lat'] * current['load'] + neighbor['lat'] * neighbor['load']) / temp_total_load
                temp_lng = (current['lng'] * current['load'] + neighbor['lng'] * neighbor['load']) / temp_total_load

                # 2. 预估新半径 (严格包络)
                d_to_curr = haversine_vectorized(temp_lng, temp_lat, current['lng'], current['lat'])
                d_to_neigh = haversine_vectorized(temp_lng, temp_lat, neighbor['lng'], neighbor['lat'])

                strict_new_radius = max(d_to_curr + current['radius'], d_to_neigh + neighbor['radius'])

                # 3. 检查半径是否超标 (给一点点宽容度，比如 1.05倍，防止因为几米误差导致无法合并)
                if strict_new_radius > radius_limit * 1.05:
                    continue

                if dist < min_dist:
                    min_dist = dist
                    best_merge_idx = j
                    # 暂存计算好的新属性，避免下面重复算
                    best_merge_props = (temp_lat, temp_lng, strict_new_radius)

            if best_merge_idx != -1:
                neighbor = centers[best_merge_idx]

                # 取出暂存的属性
                new_lat, new_lng, new_radius = best_merge_props

                # 更新
                current['lat'] = new_lat
                current['lng'] = new_lng
                current['radius'] = new_radius  # 使用严格半径
                current['load'] += neighbor['load']
                current['capacity_rate'] = current['load'] / MAX_CAPACITY
                current['center_sales'] += neighbor['center_sales']

                active_indices.remove(best_merge_idx)

            new_centers_list.append(current)

        self.final_centers = new_centers_list

    def solve(self):
        self.final_centers = []
        self.shop_assignments = {}

        # 1. 初始生成
        cities = self.df[COL_CITY].unique()
        for city in cities:
            self.process_city(city)

        # 2. 循环优化 (Iterative Optimization)
        print("开始循环优化...")
        max_iterations = 5  # 最多跑5轮，防止死循环

        for i in range(max_iterations):
            # 记录优化前的站点数量
            count_before = len(self.final_centers)

            # 第一步：吃甜甜圈
            self.post_process_absorb()

            # 第二步：合并邻居
            self.post_process_merge_neighbors()

            # 记录优化后的站点数量
            count_after = len(self.final_centers)

            print(f"优化轮次 {i + 1}: 站点数 {count_before} -> {count_after}")

            # 如果这一轮没有减少任何站点，说明稳定了，可以提前退出
            if count_after == count_before:
                print("优化已收敛，停止迭代。")
                break

        # --- 整理并输出结果 ---
        centers_df = pd.DataFrame(self.final_centers)

        # 回写结果
        result_df = self.df.copy()
        result_df['is_covered'] = False
        result_df['center_id'] = None
        result_df['distance_to_center'] = 0.0

        assignment_df = pd.DataFrame.from_dict(self.shop_assignments, orient='index')

        if not assignment_df.empty:
            # 确保索引类型一致
            result_df.loc[assignment_df.index, 'center_id'] = assignment_df['center_id']
            result_df.loc[assignment_df.index, 'distance_to_center'] = assignment_df['distance']
            result_df.loc[assignment_df.index, 'is_covered'] = True

        return centers_df, result_df
