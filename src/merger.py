import numpy as np
from sklearn.neighbors import BallTree


class CandidateMerger:
    """
    负责处理多源召回点位的融合与去重。
    [优化版] 使用 BallTree 进行空间抑制，大幅提升处理万级点位的速度。
    """

    @staticmethod
    def merge_and_prune(candidates_120, candidates_50, distance_threshold_km=0.5):
        print(f"[Merger] 启动空间抑制融合: 阈值={distance_threshold_km}km")

        final_candidates = []

        # 1. 无条件保留所有 50 点位
        coords_50 = []
        for c in candidates_50:
            new_c = c.copy()
            new_c['source_type'] = '50_high_precision'
            final_candidates.append(new_c)
            coords_50.append([c['lat'], c['lng']])

        if not coords_50:
            # 如果没有50点位，直接返回120点位
            return candidates_120

        # 2. 构建 50 点位的空间索引
        # BallTree 需要弧度坐标
        coords_50_rad = np.radians(coords_50)
        tree = BallTree(coords_50_rad, metric='haversine')

        # 转换阈值为弧度
        radius_rad = distance_threshold_km / 6371.0

        # 3. 批量处理 120 点位
        # 提取 120 点位的坐标
        coords_120 = [[c['lat'], c['lng']] for c in candidates_120]
        if not coords_120:
            return final_candidates

        coords_120_rad = np.radians(coords_120)

        # 核心优化：一次性查询所有 120 点位在阈值范围内是否有 50 点位
        # query_radius 返回的是一个数组，每个元素是 list of indices
        # count_only=True 更快，我们只需要知道有没有，不需要知道是哪个
        counts = tree.query_radius(coords_120_rad, r=radius_rad, count_only=True)

        kept_count = 0
        removed_count = 0

        for i, count in enumerate(counts):
            if count > 0:
                # 范围内有 50 点位 -> 冗余 -> 剔除
                removed_count += 1
            else:
                # 范围内无 50 点位 -> 互补 -> 保留
                new_c = candidates_120[i].copy()
                new_c['source_type'] = '120_supplement'
                final_candidates.append(new_c)
                kept_count += 1

        print(f"[Merger] 融合完成: 保留50点位 {len(candidates_50)} 个")
        print(f"[Merger] 120点位处理: 保留 {kept_count} 个 (互补), 剔除 {removed_count} 个 (距离过近)")
        print(f"[Merger] 最终初始点位总数: {len(final_candidates)}")

        return final_candidates
