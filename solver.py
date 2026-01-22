# solver.py (100%覆盖 + 极致优化版)
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from scipy.optimize import minimize
from config import *
from utils import haversine_vectorized


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

    def recalculate_geometry(self, shop_indices):
        """
        [优化] 计算最小覆盖圆，并增加微小缓冲确保覆盖
        """
        if not shop_indices:
            return 0.0, 0.0, 0.1

        subset = self.df.loc[shop_indices]
        lats = subset[COL_LAT].values
        lngs = subset[COL_LNG].values

        # 1. 初始重心
        init_lat = lats.mean()
        init_lng = lngs.mean()

        if len(shop_indices) <= 2:
            dists = haversine_vectorized(init_lng, init_lat, lngs, lats)
            return init_lat, init_lng, max(dists.max() + 0.001, 0.1)  # +1米缓冲

        # 2. 优化目标
        def objective_function(center_coord):
            c_lat, c_lng = center_coord
            dists = haversine_vectorized(c_lng, c_lat, lngs, lats)
            return dists.max()

        try:
            # 限制迭代次数，平衡速度与精度
            result = minimize(
                objective_function,
                x0=[init_lat, init_lng],
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-3}
            )
            best_lat, best_lng = result.x
        except:
            best_lat, best_lng = init_lat, init_lng

        # 3. [关键] 重新计算精确半径并加缓冲
        # 优化器返回的 result.fun 可能是近似值，必须重新算
        final_dists = haversine_vectorized(best_lng, best_lat, lngs, lats)
        final_radius = final_dists.max() + 0.001  # 增加 1米 缓冲，防止精度误差导致覆盖判定失败

        return best_lat, best_lng, max(final_radius, 0.1)

    def save_cluster(self, df_subset, city_name, city_tier, center_lat, center_lng, radius):
        center_id = f"Auto_{len(self.final_centers) + 1}"
        shop_indices = df_subset.index.tolist()

        self.final_centers.append({
            'center_id': center_id,
            'city': city_name,
            'city_tier': city_tier,
            'lat': center_lat,
            'lng': center_lng,
            'radius': radius,
            'load': len(df_subset),
            'capacity_rate': len(df_subset) / MAX_CAPACITY,
            'center_sales': df_subset[COL_SALES].sum() if COL_SALES in df_subset.columns else 0,
            'shop_indices': shop_indices
        })

    def process_city(self, city_name):
        """
        标准生成流程 (移除激进的半径压缩，保证初始覆盖)
        """
        city_df = self.df[self.df[COL_CITY] == city_name].copy()
        radius_limit_km, city_tier = self.get_radius_limit(city_df)
        print(f"处理 {city_name} ({city_tier})...")

        coords_rad = np.radians(city_df[[COL_LAT, COL_LNG]].values)
        tree = BallTree(coords_rad, metric='haversine')

        processed_indices = set()
        all_indices = city_df.index.to_numpy()

        # 密度优先
        density_counts = tree.query_radius(coords_rad, r=2.0 / 6371.0, count_only=True)
        sorted_indices = np.argsort(-density_counts)

        R = 6371.0
        radius_limit_rad = radius_limit_km / R

        # 禁区列表 (适度排斥)
        forbidden_zones = []

        for i in sorted_indices:
            original_idx = all_indices[i]
            if original_idx in processed_indices: continue

            seed_lat = city_df.loc[original_idx, COL_LAT]
            seed_lng = city_df.loc[original_idx, COL_LNG]

            # 适度排斥：只检查非常近的距离 (0.5倍半径)
            # 既减少重合，又不至于产生无法填充的缝隙
            in_forbidden = False
            for fz_lat, fz_lng, fz_radius in forbidden_zones:
                dist = haversine_vectorized(fz_lng, fz_lat, seed_lng, seed_lat)
                if dist < max(fz_radius * 0.5, 0.5):
                    in_forbidden = True
                    break
            if in_forbidden: continue

            seed_coord = coords_rad[i].reshape(1, -1)
            k_query = min(len(city_df), MAX_CAPACITY * 3)
            dist_rad, tree_indices = tree.query(seed_coord, k=k_query)
            dist_rad, tree_indices = dist_rad[0], tree_indices[0]

            cluster_indices = []
            for d_rad, t_idx in zip(dist_rad, tree_indices):
                real_idx = all_indices[t_idx]
                if real_idx in processed_indices: continue
                if d_rad > radius_limit_rad: break
                cluster_indices.append(real_idx)
                if len(cluster_indices) >= MAX_CAPACITY: break

            if cluster_indices:
                subset = city_df.loc[cluster_indices]
                center_lat, center_lng, final_radius = self.recalculate_geometry(cluster_indices)

                self.save_cluster(subset, city_name, city_tier, center_lat, center_lng, final_radius)
                processed_indices.update(cluster_indices)
                forbidden_zones.append((center_lat, center_lng, final_radius))

    def post_process_absorb(self):
        """吞噬优化"""
        centers = self.final_centers
        centers.sort(key=lambda x: x['radius'], reverse=True)
        active_indices = set(range(len(centers)))
        new_centers_list = []

        for i in range(len(centers)):
            if i not in active_indices: continue
            big = centers[i]
            merged_indices = []

            for j in range(len(centers)):
                if i == j or j not in active_indices: continue
                small = centers[j]
                if big['city'] != small['city']: continue

                dist = haversine_vectorized(big['lng'], big['lat'], small['lng'], small['lat'])
                if dist + small['radius'] > big['radius'] * 1.3: continue

                if big['load'] + small['load'] <= MAX_CAPACITY:
                    combined = big['shop_indices'] + small['shop_indices']
                    n_lat, n_lng, n_rad = self.recalculate_geometry(combined)

                    if n_rad <= big['radius'] * 1.15:  # 稍微放宽吞噬条件
                        big['lat'], big['lng'], big['radius'] = n_lat, n_lng, n_rad
                        big['load'] += small['load']
                        big['capacity_rate'] = big['load'] / MAX_CAPACITY
                        big['center_sales'] += small['center_sales']
                        big['shop_indices'] = combined
                        merged_indices.append(j)

            for idx in merged_indices: active_indices.remove(idx)
            if i in active_indices: new_centers_list.append(big)
        self.final_centers = new_centers_list

    def post_process_merge_neighbors(self):
        """优先合并重合度高的邻居"""
        centers = self.final_centers
        active_indices = set(range(len(centers)))
        new_centers_list = []

        for i in range(len(centers)):
            if i not in active_indices: continue
            current = centers[i]
            best_merge_idx = -1
            max_score = -99999.0

            tier = current.get('city_tier', '未知')
            limit = TIER_RADIUS_LIMIT.get(tier, DEFAULT_RADIUS_LIMIT)

            for j in range(len(centers)):
                if i == j or j not in active_indices: continue
                neighbor = centers[j]
                if current['city'] != neighbor['city']: continue
                if current['load'] + neighbor['load'] > MAX_CAPACITY: continue

                dist = haversine_vectorized(current['lng'], current['lat'], neighbor['lng'], neighbor['lat'])

                # 快速初筛
                if (current['radius'] + neighbor['radius'] + dist) / 2 > limit * 1.2: continue

                overlap = (current['radius'] + neighbor['radius']) - dist
                score = overlap + 1000 if overlap > 0 else -dist

                if score > max_score:
                    combined = current['shop_indices'] + neighbor['shop_indices']
                    n_lat, n_lng, n_rad = self.recalculate_geometry(combined)

                    if n_rad <= limit * 1.05:
                        max_score = score
                        best_merge_idx = j
                        best_props = (n_lat, n_lng, n_rad, combined)

            if best_merge_idx != -1:
                neighbor = centers[best_merge_idx]
                current['lat'], current['lng'], current['radius'] = best_props[0], best_props[1], best_props[2]
                current['load'] += neighbor['load']
                current['capacity_rate'] = current['load'] / MAX_CAPACITY
                current['center_sales'] += neighbor['center_sales']
                current['shop_indices'] = best_props[3]
                active_indices.remove(best_merge_idx)

            new_centers_list.append(current)
        self.final_centers = new_centers_list

    def post_process_merge_small_sites(self):
        """
        [关键] 强制清理小站点
        允许半径放宽到 1.1 倍，以减少点位数量
        """
        global best_props
        print("清理低负载站点...")
        centers = self.final_centers
        centers.sort(key=lambda x: x['load'])  # 从小到大处理

        active_indices = set(range(len(centers)))
        new_centers_list = []

        for i in range(len(centers)):
            if i not in active_indices: continue
            current = centers[i]

            # 如果负载还行(>60%)，就不强制合并了
            if current['capacity_rate'] > 0.6:
                new_centers_list.append(current)
                continue

            best_idx = -1
            min_dist = float('inf')
            tier = current.get('city_tier', '未知')
            limit = TIER_RADIUS_LIMIT.get(tier, DEFAULT_RADIUS_LIMIT) * 1.1  # 放宽限制

            for j in range(len(centers)):
                if i == j or j not in active_indices: continue
                neighbor = centers[j]
                if current['city'] != neighbor['city']: continue
                if current['load'] + neighbor['load'] > MAX_CAPACITY: continue

                dist = haversine_vectorized(current['lng'], current['lat'], neighbor['lng'], neighbor['lat'])
                if dist > limit * 2:
                    continue  # 太远不看

                combined = current['shop_indices'] + neighbor['shop_indices']
                n_lat, n_lng, n_rad = self.recalculate_geometry(combined)

                if n_rad <= limit:
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = j
                        best_props = (n_lat, n_lng, n_rad, combined)

            if best_idx != -1:
                neighbor = centers[best_idx]
                neighbor['lat'], neighbor['lng'], neighbor['radius'] = best_props[0], best_props[1], best_props[2]
                neighbor['load'] += current['load']
                neighbor['capacity_rate'] = neighbor['load'] / MAX_CAPACITY
                neighbor['center_sales'] += current['center_sales']
                neighbor['shop_indices'] = best_props[3]
                # current 被合并了，不加入 new list
            else:
                new_centers_list.append(current)

        self.final_centers = new_centers_list

    def post_process_ensure_coverage(self):
        """
        [兜底] 确保 100% 覆盖
        扫描所有店铺，如果有漏网之鱼，强制塞给最近的站点
        """
        print("执行最终覆盖检查...")
        covered_shops = set()
        for c in self.final_centers:
            covered_shops.update(c['shop_indices'])

        all_shops = set(self.df.index)
        orphans = list(all_shops - covered_shops)

        if not orphans:
            print("  ✅ 完美覆盖 (100%)")
            return

        print(f"  ⚠️ 发现 {len(orphans)} 个孤儿店铺，正在强制分配...")

        # 建立站点索引
        center_coords = np.radians([[c['lat'], c['lng']] for c in self.final_centers])
        tree = BallTree(center_coords, metric='haversine')

        for oid in orphans:
            o_row = self.df.loc[oid]
            o_coord = np.radians([[o_row[COL_LAT], o_row[COL_LNG]]])

            # 找最近的站点
            dist, ind = tree.query(o_coord, k=5)

            assigned = False
            # 优先给同城且未满的
            for d, idx in zip(dist[0], ind[0]):
                c = self.final_centers[idx]
                if c['city'] == o_row[COL_CITY] and c['load'] < MAX_CAPACITY:
                    c['shop_indices'].append(oid)
                    c['load'] += 1
                    assigned = True
                    break

            # 如果都满了，强制给最近的同城站点 (允许超载)
            if not assigned:
                for idx in ind[0]:
                    c = self.final_centers[idx]
                    if c['city'] == o_row[COL_CITY]:
                        c['shop_indices'].append(oid)
                        c['load'] += 1
                        assigned = True
                        break

            # 极端情况：新建站点
            if not assigned:
                self.save_cluster(self.df.loc[[oid]], o_row[COL_CITY], '未知', o_row[COL_LAT], o_row[COL_LNG], 0.1)

    def solve(self):
        self.final_centers = []
        self.shop_assignments = {}

        # 1. 初始生成
        cities = self.df[COL_CITY].unique()
        for city in cities:
            self.process_city(city)

        # 2. 循环优化
        print("开始循环优化...")
        for i in range(5):
            count_before = len(self.final_centers)
            self.post_process_absorb()
            self.post_process_merge_neighbors()
            self.post_process_merge_small_sites()  # 每一轮都尝试清理小站点

            count_after = len(self.final_centers)
            print(f"轮次 {i + 1}: {count_before} -> {count_after}")
            if count_after == count_before: break

        # 3. [兜底] 确保覆盖
        self.post_process_ensure_coverage()

        # 4. 最终几何重算
        for c in self.final_centers:
            if c['load'] > 0:
                n_lat, n_lng, n_rad = self.recalculate_geometry(c['shop_indices'])
                c['lat'], c['lng'], c['radius'] = n_lat, n_lng, n_rad
                c['center_sales'] = self.df.loc[c['shop_indices']][COL_SALES].sum()

        # 5. 输出
        centers_df = pd.DataFrame(self.final_centers)
        for c in self.final_centers:
            c_id = c['center_id']
            subset = self.df.loc[c['shop_indices']]
            dists = haversine_vectorized(c['lng'], c['lat'], subset[COL_LNG].values, subset[COL_LAT].values)
            for idx, dist in zip(subset.index, dists):
                self.shop_assignments[idx] = {'center_id': c_id, 'distance': dist}

        result_df = self.df.copy()
        result_df['is_covered'] = False
        result_df['center_id'] = None
        result_df['distance_to_center'] = 0.0

        assignment_df = pd.DataFrame.from_dict(self.shop_assignments, orient='index')
        if not assignment_df.empty:
            result_df.loc[assignment_df.index, 'center_id'] = assignment_df['center_id']
            result_df.loc[assignment_df.index, 'distance_to_center'] = assignment_df['distance']
            result_df.loc[assignment_df.index, 'is_covered'] = True

        if 'shop_indices' in centers_df.columns:
            centers_df = centers_df.drop(columns=['shop_indices'])

        return centers_df, result_df
