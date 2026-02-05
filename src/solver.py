import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from scipy.optimize import minimize
from src.config import *
from src.utils import haversine_vectorized, fast_min_enclosing_circle


class CoverageSolver:
    # 1. 修改：增加 max_capacity 参数，默认值为 None
    def __init__(self, df, max_capacity=None):
        self.df = df.copy()

        # 2. 修改：如果有传入参数就用参数，否则用 config 里的全局默认值
        self.max_capacity = max_capacity if max_capacity is not None else MAX_CAPACITY

        # --- 数据清洗逻辑 ---
        self.df[COL_LNG] = pd.to_numeric(self.df[COL_LNG], errors='coerce')
        self.df[COL_LAT] = pd.to_numeric(self.df[COL_LAT], errors='coerce')

        initial_count = len(self.df)
        self.df = self.df.dropna(subset=[COL_LNG, COL_LAT])

        dropped_count = initial_count - len(self.df)
        if dropped_count > 0:
            print(f"⚠️ 警告: 发现并移除了 {dropped_count} 条经纬度无效的数据！")

        self.df[COL_LNG] = self.df[COL_LNG].astype(float)
        self.df[COL_LAT] = self.df[COL_LAT].astype(float)

        self.final_centers = []
        self.shop_assignments = {}

    # --- 新增：加载外部初始解 ---
    def load_external_candidates(self, candidates):
        """
        将外部融合好的点位列表加载为 Solver 的初始解。
        """
        self.final_centers = []
        print(f"[Solver] 正在加载 {len(candidates)} 个外部初始点位...")

        for i, c in enumerate(candidates):
            shop_indices = c.get('shop_indices', [])
            if not shop_indices: continue

            # 尝试获取城市
            try:
                first_shop_idx = shop_indices[0]
                city_name = self.df.loc[first_shop_idx, COL_CITY]
            except KeyError:
                continue

                # 重新计算几何
            lat, lng, radius = self.recalculate_geometry(shop_indices)

            # 获取城市等级
            city_df = self.df[self.df[COL_CITY] == city_name]
            if city_df.empty:
                tier = '未知'
            else:
                _, tier = self.get_radius_limit(city_df)

            # 计算销量
            center_sales = 0
            if COL_SALES and COL_SALES in self.df.columns:
                center_sales = self.df.loc[shop_indices][COL_SALES].sum()

            self.final_centers.append({
                'center_id': c.get('center_id', f"Init_{i}"),
                'city': city_name,
                'city_tier': tier,
                'lat': lat,
                'lng': lng,
                'radius': radius,
                'load': len(shop_indices),
                'capacity_rate': len(shop_indices) / self.max_capacity,  # <--- 替换
                'center_sales': center_sales,
                'shop_indices': shop_indices,
                'source_type': c.get('source_type', 'external')
            })

        print(f"[Solver] 成功加载 {len(self.final_centers)} 个有效初始点位。")

    @staticmethod
    def get_radius_limit(city_df):
        try:
            tier = city_df.iloc[0][COL_CITY_TIER]
        except (KeyError, IndexError):
            tier = '未知'
        return TIER_RADIUS_LIMIT.get(tier, DEFAULT_RADIUS_LIMIT), tier

    def recalculate_geometry(self, shop_indices):
        if not shop_indices:
            return 0.0, 0.0, 0.1

        # 直接从 self.df 提取 numpy array，避免多次 loc 开销
        # 注意：这里假设 shop_indices 是 list
        subset = self.df.loc[shop_indices]
        lats = subset[COL_LAT].values
        lngs = subset[COL_LNG].values

        # 调用新的快速算法
        return fast_min_enclosing_circle(lats, lngs)

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
            'capacity_rate': len(df_subset) / self.max_capacity,  # <--- 替换
            'center_sales': df_subset[COL_SALES].sum() if COL_SALES in df_subset.columns else 0,
            'shop_indices': shop_indices
        })

    def process_city(self, city_name):
        city_df = self.df[self.df[COL_CITY] == city_name].copy()
        radius_limit_km, city_tier = self.get_radius_limit(city_df)
        print(f"处理 {city_name} ({city_tier})...")

        coords_rad = np.radians(city_df[[COL_LAT, COL_LNG]].values)
        tree = BallTree(coords_rad)

        processed_indices = set()
        all_indices = city_df.index.to_numpy()

        density_counts = tree.query_radius(coords_rad, r=2.0 / 6371.0, count_only=True)
        sorted_indices = np.argsort(-density_counts)

        R = 6371.0
        radius_limit_rad = radius_limit_km / R

        forbidden_zones = []

        for i in sorted_indices:
            original_idx = all_indices[i]
            if original_idx in processed_indices: continue

            seed_lat = city_df.loc[original_idx, COL_LAT]
            seed_lng = city_df.loc[original_idx, COL_LNG]

            in_forbidden = False
            for fz_lat, fz_lng, fz_radius in forbidden_zones:
                dist = haversine_vectorized(fz_lng, fz_lat, seed_lng, seed_lat)
                if dist < max(fz_radius * 0.5, 0.5):
                    in_forbidden = True
                    break
            if in_forbidden: continue

            seed_coord = coords_rad[i].reshape(1, -1)
            # <--- 替换：使用 self.max_capacity
            k_query = min(len(city_df), self.max_capacity * 3)
            dist_rad, tree_indices = tree.query(seed_coord, k=k_query)
            dist_rad, tree_indices = dist_rad[0], tree_indices[0]

            cluster_indices = []
            for d_rad, t_idx in zip(dist_rad, tree_indices):
                real_idx = all_indices[t_idx]
                if real_idx in processed_indices: continue
                if d_rad > radius_limit_rad: break
                cluster_indices.append(real_idx)
                # <--- 替换：使用 self.max_capacity
                if len(cluster_indices) >= self.max_capacity: break

            if cluster_indices:
                subset = city_df.loc[cluster_indices]
                center_lat, center_lng, final_radius = self.recalculate_geometry(cluster_indices)

                self.save_cluster(subset, city_name, city_tier, center_lat, center_lng, final_radius)
                processed_indices.update(cluster_indices)
                forbidden_zones.append((center_lat, center_lng, final_radius))

    def post_process_absorb(self):
        """
        [优化版] 吞噬优化：大站吃小站
        使用 BallTree 加速邻域搜索
        """
        centers = self.final_centers
        if not centers: return

        # 按半径降序，优先保留大站
        centers.sort(key=lambda x: x['radius'], reverse=True)

        # 建立状态标记，避免列表频繁删除
        active_mask = np.ones(len(centers), dtype=bool)

        # 构建空间索引 (以弧度为单位)
        coords = np.radians([[c['lat'], c['lng']] for c in centers])
        tree = BallTree(coords, metric='haversine')

        # 预计算所有站点的最大可能搜索半径 (例如最大限制半径的2倍)
        # 这里为了安全，取全局最大限制 (例如 4km) 的 1.5 倍作为搜索域
        # 两个圆心距离如果超过 R_limit + R_small，就不可能合并。
        # 粗略给一个 10km 的搜索范围足够覆盖绝大多数合并可能
        search_radius_rad = 10.0 / 6371.0

        for i in range(len(centers)):
            if not active_mask[i]: continue

            big = centers[i]
            tier = big.get('city_tier', '未知')
            limit = TIER_RADIUS_LIMIT.get(tier, DEFAULT_RADIUS_LIMIT)

            # 查询附近的潜在邻居
            # query_radius 返回的是索引数组
            indices = tree.query_radius([coords[i]], r=search_radius_rad)[0]

            merged_indices = []

            for j in indices:
                if i == j or not active_mask[j]: continue

                small = centers[j]

                # 基础过滤
                if big['city'] != small['city']: continue
                if big['load'] + small['load'] > self.max_capacity: continue

                # 精确距离判断
                dist = haversine_vectorized(big['lng'], big['lat'], small['lng'], small['lat'])

                # 核心几何约束：合并后半径不能超标
                # 快速预判：如果两圆心距离 + 小圆半径 已经超过限制，则不必计算几何
                if dist + small['radius'] > limit: continue

                # 尝试合并几何计算
                combined = big['shop_indices'] + small['shop_indices']
                n_lat, n_lng, n_rad = self.recalculate_geometry(combined)

                if n_rad <= limit:
                    # 执行吞噬
                    big['lat'], big['lng'], big['radius'] = n_lat, n_lng, n_rad
                    big['load'] += small['load']
                    big['capacity_rate'] = big['load'] / self.max_capacity
                    big['center_sales'] += small['center_sales']
                    big['shop_indices'] = combined

                    # 标记被吃掉的站点
                    active_mask[j] = False
                    merged_indices.append(j)

            # 注意：这里不需要移除 merged_indices，因为 active_mask 已经处理了

        # 重建列表
        self.final_centers = [c for i, c in enumerate(centers) if active_mask[i]]

    def post_process_merge_neighbors(self):
        """
        [优化版] 邻居合并：两个邻居合并成一个新的
        使用 BallTree 加速
        """
        centers = self.final_centers
        if not centers: return

        # 这里的顺序不重要，但为了确定性可以按 ID 或负载排序
        # active_mask 逻辑同上
        active_mask = np.ones(len(centers), dtype=bool)

        coords = np.radians([[c['lat'], c['lng']] for c in centers])
        tree = BallTree(coords, metric='haversine')

        search_radius_rad = 10.0 / 6371.0

        for i in range(len(centers)):
            if not active_mask[i]: continue

            current = centers[i]
            tier = current.get('city_tier', '未知')
            limit = TIER_RADIUS_LIMIT.get(tier, DEFAULT_RADIUS_LIMIT)

            best_merge_idx = -1
            max_score = -99999.0
            best_props_combined = None

            # 查询邻居
            indices = tree.query_radius([coords[i]], r=search_radius_rad)[0]

            for j in indices:
                if i == j or not active_mask[j]: continue
                neighbor = centers[j]

                if current['city'] != neighbor['city']: continue
                if current['load'] + neighbor['load'] > self.max_capacity: continue

                dist = haversine_vectorized(current['lng'], current['lat'], neighbor['lng'], neighbor['lat'])

                # 快速剪枝：如果两个圆相距太远，合并后的圆肯定很大
                if (current['radius'] + neighbor['radius'] + dist) / 2 > limit: continue

                # 计算重叠度分数 (同原逻辑)
                overlap = (current['radius'] + neighbor['radius']) - dist
                score = overlap + 1000 if overlap > 0 else -dist

                if score > max_score:
                    combined = current['shop_indices'] + neighbor['shop_indices']
                    n_lat, n_lng, n_rad = self.recalculate_geometry(combined)

                    if n_rad <= limit:
                        max_score = score
                        best_merge_idx = j
                        best_props_combined = (n_lat, n_lng, n_rad, combined)

            if best_merge_idx != -1:
                # 执行合并：更新 current，标记 neighbor 删除
                neighbor_idx = best_merge_idx
                neighbor = centers[neighbor_idx]

                current['lat'], current['lng'], current['radius'] = best_props_combined[0], best_props_combined[1], \
                best_props_combined[2]
                current['load'] += neighbor['load']
                current['capacity_rate'] = current['load'] / self.max_capacity
                current['center_sales'] += neighbor['center_sales']
                current['shop_indices'] = best_props_combined[3]

                active_mask[neighbor_idx] = False

        self.final_centers = [c for i, c in enumerate(centers) if active_mask[i]]

    def post_process_merge_small_sites(self):
        """
        [优化版] 清理小站点：尝试将低负载站点合并入附近的站点
        使用 BallTree 加速搜索
        """
        centers = self.final_centers
        if not centers: return

        # 按负载从小到大排序，优先处理最“穷”的站点
        centers.sort(key=lambda x: x['load'])

        active_mask = np.ones(len(centers), dtype=bool)

        # 构建空间索引
        coords = np.radians([[c['lat'], c['lng']] for c in centers])
        tree = BallTree(coords, metric='haversine')

        # 搜索半径：如果距离超过 max_limit，合并后的圆基本不可能满足半径约束
        # 这里取一个宽松值，比如 5km
        search_radius_rad = 5.0 / 6371.0

        for i in range(len(centers)):
            if not active_mask[i]: continue

            current = centers[i]

            # 如果负载已经很高（例如 > 60%），就不折腾了，保留
            if current['capacity_rate'] > 0.6:
                continue

            best_idx = -1
            min_dist = float('inf')
            best_props = None

            tier = current.get('city_tier', '未知')
            limit = TIER_RADIUS_LIMIT.get(tier, DEFAULT_RADIUS_LIMIT)

            # 1. 空间查询：只找附近的点
            indices = tree.query_radius([coords[i]], r=search_radius_rad)[0]

            for j in indices:
                # 排除自己，排除已删除的点
                if i == j or not active_mask[j]: continue

                neighbor = centers[j]

                # 基础约束检查
                if current['city'] != neighbor['city']: continue
                if current['load'] + neighbor['load'] > self.max_capacity: continue

                dist = haversine_vectorized(current['lng'], current['lat'], neighbor['lng'], neighbor['lat'])

                # 如果比当前找到的最优解还远，跳过
                if dist >= min_dist: continue

                # 几何试算
                combined = current['shop_indices'] + neighbor['shop_indices']
                n_lat, n_lng, n_rad = self.recalculate_geometry(combined)

                if n_rad <= limit:
                    # 找到了一个合法的合并对象
                    min_dist = dist
                    best_idx = j
                    best_props = (n_lat, n_lng, n_rad, combined)

            if best_idx != -1:
                # 执行合并：将 current 并入 neighbor
                # 注意：这里我们更新 neighbor，标记 current 为删除
                neighbor = centers[best_idx]
                neighbor['lat'], neighbor['lng'], neighbor['radius'] = best_props[0], best_props[1], best_props[2]
                neighbor['load'] += current['load']
                neighbor['capacity_rate'] = neighbor['load'] / self.max_capacity
                neighbor['center_sales'] += current['center_sales']
                neighbor['shop_indices'] = best_props[3]

                # 标记当前小站被移除
                active_mask[i] = False

                # 注意：这里不需要更新 tree，虽然 neighbor 的位置变了一点点，
                # 但对于后续的搜索影响微乎其微，重建 tree 代价太大。

        # 重建列表
        self.final_centers = [c for i, c in enumerate(centers) if active_mask[i]]

    def post_process_ensure_coverage(self):
        """
        [优化版] 兜底覆盖：
        1. 尝试将孤儿塞入现有站点。
        2. 塞不进去的孤儿，不再单独建站，而是进行"二次聚类"。
        """
        print("执行最终覆盖检查 (智能聚类模式)...")

        # 1. 找出所有未覆盖店铺
        covered_shops = set()
        for c in self.final_centers:
            covered_shops.update(c['shop_indices'])

        all_shops = set(self.df.index)
        orphans = list(all_shops - covered_shops)

        if not orphans:
            print("  ✅ 完美覆盖 (100%)")
            return

        print(f"  ⚠️ 发现 {len(orphans)} 个孤儿店铺，尝试归并...")

        # 2. 第一轮：尝试塞入现有站点 (同原有逻辑，使用 BallTree 加速)
        # 构建现有站点的索引
        center_coords = np.radians([[c['lat'], c['lng']] for c in self.final_centers])
        tree = BallTree(center_coords, metric='haversine')

        remaining_orphans = []

        for oid in orphans:
            o_row = self.df.loc[oid]
            o_lat, o_lng = o_row[COL_LAT], o_row[COL_LNG]
            o_coord = np.radians([[o_lat, o_lng]])

            tier = self.get_radius_limit(self.df[self.df[COL_CITY] == o_row[COL_CITY]])[1]
            limit = TIER_RADIUS_LIMIT.get(tier, DEFAULT_RADIUS_LIMIT)

            # 找最近的 5 个站点尝试
            dist_rad, ind = tree.query(o_coord, k=min(5, len(self.final_centers)))
            assigned = False

            for idx in ind[0]:
                c = self.final_centers[idx]

                if c['city'] != o_row[COL_CITY]: continue
                if c['load'] >= self.max_capacity: continue

                # 几何校验
                temp_indices = c['shop_indices'] + [oid]
                n_lat, n_lng, n_rad = self.recalculate_geometry(temp_indices)

                if n_rad <= limit:
                    # 成功塞入
                    c['lat'], c['lng'], c['radius'] = n_lat, n_lng, n_rad
                    c['shop_indices'] = temp_indices
                    c['load'] += 1
                    c['capacity_rate'] = c['load'] / self.max_capacity
                    if COL_SALES in self.df.columns:
                        c['center_sales'] += o_row[COL_SALES]
                    assigned = True
                    break

            if not assigned:
                remaining_orphans.append(oid)

        if not remaining_orphans:
            print("  ✅ 所有孤儿已成功归并入现有站点。")
            return

        # 3. 第二轮：对剩余孤儿进行"二次聚类"
        # 逻辑：把这些孤儿当成一个新的微型城市，重新跑一遍 process_city 的核心逻辑
        print(f"  🔄 对剩余 {len(remaining_orphans)} 个孤儿进行二次聚类...")

        # 提取孤儿的 DataFrame
        orphan_df = self.df.loc[remaining_orphans].copy()

        # 按城市分组处理
        for city_name, group in orphan_df.groupby(COL_CITY):
            # 复用 process_city 的逻辑，但只针对这部分数据
            # 这里我们手动实现一个简化版的贪心聚类，避免递归调用整个 process_city 导致复杂化
            self._cluster_orphans_greedy(group, city_name)

    def _cluster_orphans_greedy(self, orphan_df, city_name):
        """
        针对孤儿的简化版贪心聚类
        """
        if orphan_df.empty: return

        # 获取半径限制
        radius_limit_km, city_tier = self.get_radius_limit(orphan_df)
        radius_limit_rad = radius_limit_km / 6371.0

        coords = np.radians(orphan_df[[COL_LAT, COL_LNG]].values)
        indices = orphan_df.index.to_numpy()

        # 建立索引
        tree = BallTree(coords, metric='haversine')

        processed = set()

        # 简单策略：遍历每个点，如果没处理过，就以它为中心画个圈
        for i in range(len(coords)):
            if indices[i] in processed: continue

            # 以当前孤儿为种子，找半径内的所有兄弟
            # 注意：这里可以直接找 max_capacity 个，因为是最后兜底
            idx_in_radius = tree.query_radius([coords[i]], r=radius_limit_rad)[0]

            cluster_indices = []
            for j in idx_in_radius:
                real_idx = indices[j]
                if real_idx in processed: continue
                cluster_indices.append(real_idx)
                if len(cluster_indices) >= self.max_capacity: break

            if cluster_indices:
                # 生成新站点
                subset = orphan_df.loc[cluster_indices]
                c_lat, c_lng, c_rad = self.recalculate_geometry(cluster_indices)

                # 保存
                self.save_cluster(subset, city_name, city_tier, c_lat, c_lng, c_rad)

                # 标记已处理
                processed.update(cluster_indices)

    # --- solve 方法支持跳过生成 ---
    def solve(self, use_existing_init=False):
        self.shop_assignments = {}

        # 1. 初始生成
        if not use_existing_init:
            print("[Solver] 模式: 全量重新计算 (Greedy Generation)")
            self.final_centers = []
            cities = self.df[COL_CITY].unique()
            for city in cities:
                self.process_city(city)
        else:
            print("[Solver] 模式: 使用外部载入的初始解 (Skip Generation)")
            if not self.final_centers:
                print("⚠️ 警告: use_existing_init=True 但没有加载任何初始点位！将自动回退到全量计算。")
                cities = self.df[COL_CITY].unique()
                for city in cities:
                    self.process_city(city)

        # 2. 循环优化
        print("开始循环优化...")
        for i in range(5):
            count_before = len(self.final_centers)
            self.post_process_absorb()
            self.post_process_merge_neighbors()
            self.post_process_merge_small_sites()

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
                c['capacity_rate'] = c['load'] / self.max_capacity  # <--- 替换
                if COL_SALES and COL_SALES in self.df.columns:
                    c['center_sales'] = self.df.loc[c['shop_indices']][COL_SALES].sum()
                else:
                    c['center_sales'] = 0

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
