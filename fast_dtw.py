import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import argparse
import math
import os

class ActionMatcher:
    def __init__(self, reference_csvs=None, threshold=1000, oks_sigma=0.1, min_confidence=0.3):
        """
        初始化动作匹配器，支持多动作标准数据集
        
        参数:
            reference_csvs: 标准数据集CSV文件路径字典 {动作编号: 文件路径}
            threshold: 默认匹配阈值
            oks_sigma: 默认OKS算法的sigma参数
            min_confidence: 默认关键点最小置信度阈值
        """
        self.reference_data = {}
        self.reference_labels = {}
        self.matched_indices = {}
        self.matched_distances = {}
        
        # 默认参数
        self.default_threshold = threshold
        self.default_oks_sigma = oks_sigma 
        self.default_min_confidence = min_confidence
        
        # 设置实例参数
        self.threshold = threshold
        self.oks_sigma = oks_sigma
        self.min_confidence = min_confidence
        
        # 各动作的自定义参数 - 更新为完整的动作映射
        self.action_params = {
            0: {"threshold": 3.5, "oks_sigma": 0.25, "min_confidence": 0.15},  # forward_stance_low_block
            1: {"threshold": 3.0, "oks_sigma": 0.2, "min_confidence": 0.2},   # forward_stance_punch
            2: {"threshold": 2.8, "oks_sigma": 0.18, "min_confidence": 0.2},  # front_kick_punch
            3: {"threshold": 2.8, "oks_sigma": 0.18, "min_confidence": 0.2},  # front_punch
            4: {"threshold": 2.8, "oks_sigma": 0.18, "min_confidence": 0.2},  # low_block
            5: {"threshold": 2.8, "oks_sigma": 0.18, "min_confidence": 0.2},  # middle_block
            6: {"threshold": 3.5, "oks_sigma": 0.25, "min_confidence": 0.15}, # preparation 
            7: {"threshold": 2.8, "oks_sigma": 0.18, "min_confidence": 0.2},  # rising_block
            8: {"threshold": 3.0, "oks_sigma": 0.2, "min_confidence": 0.2},   # turn_forward_stance_low_block
            9: {"threshold": 2.8, "oks_sigma": 0.18, "min_confidence": 0.2},  # turn_forward_stance_punch
        }
        
        # 加载标准数据集
        if reference_csvs:
            for action_id, csv_path in reference_csvs.items():
                self.load_reference_data(action_id, csv_path)
    
    def load_reference_data(self, action_id, csv_path):
        """加载指定动作的标准数据集"""
        try:
            if not os.path.exists(csv_path):
                print(f"警告: 标准数据文件不存在: {csv_path}")
                return False
                
            df = pd.read_csv(csv_path)
            self.reference_data[action_id] = self._preprocess_data(df)
            self.reference_labels[action_id] = df['action_label'].values if 'action_label' in df.columns else None
            self.matched_indices[action_id] = set()
            self.matched_distances[action_id] = []
            print(f"已加载动作 {action_id} 的标准数据，共 {len(self.reference_data[action_id])} 帧")
            return True
        except Exception as e:
            print(f"加载动作 {action_id} 的标准数据失败: {e}")
            return False
    
    def _preprocess_data(self, df):
        """预处理数据，提取关键点坐标和置信度，并进行归一化与异常点过滤"""
        # 提取所有关键点列
        kp_cols = [col for col in df.columns if col.startswith('kp') or 
                (col.endswith('_x') or col.endswith('_y') or col.endswith('_conf'))]
        
        if not kp_cols:
            raise ValueError("CSV文件中未找到关键点数据列")
            
        # 确定关键点数量
        if any(col.startswith('kp') for col in kp_cols):
            # 格式为kp0_x, kp0_y, kp0_conf, kp1_x, ...
            n_points = len(set(col.split('_')[0] for col in kp_cols if col.startswith('kp')))
        else:
            # 格式为x0, y0, conf0, x1, y1, conf1, ...
            n_points = len([col for col in kp_cols if col.endswith('_x')])
        
        # 提取数据
        data = []
        for _, row in df.iterrows():
            points = np.zeros((n_points, 3))  # (x, y, confidence)
            
            # 填充关键点数据
            for i in range(n_points):
                if f'kp{i}_x' in df.columns:
                    # 第一种命名格式
                    x = row[f'kp{i}_x']
                    y = row[f'kp{i}_y']
                    conf = row[f'kp{i}_conf']
                else:
                    # 第二种命名格式
                    x = row[f'x{i}'] if f'x{i}' in df.columns else row.get(f'kp{i}_x', 0)
                    y = row[f'y{i}'] if f'y{i}' in df.columns else row.get(f'kp{i}_y', 0)
                    conf = row[f'conf{i}'] if f'conf{i}' in df.columns else row.get(f'kp{i}_conf', 0)
                
                # 低置信度处理
                if conf < self.min_confidence:
                    x, y, conf = 0, 0, 0
                
                points[i] = [x, y, conf]

            # 大数值归一化
            if np.max(np.abs(points[:, :2])) > 1000:
                points[:, :2] /= 1000

            # 异常点过滤
            valid_points = points[points[:, 2] >= self.min_confidence]
            if len(valid_points) > 1:
                coords = valid_points[:, :2]
                median = np.median(coords, axis=0)
                std = np.std(coords, axis=0)
                
                # 检测离群点（3σ原则）
                outliers = np.any(np.abs(coords - median) > 3 * std, axis=1)
                if np.any(outliers):
                    outlier_indices = np.where(points[:, 2] >= self.min_confidence)[0][outliers]
                    points[outlier_indices, :] = 0

            # 多尺度归一化
            valid_points = points[points[:, 2] >= self.min_confidence]
            if len(valid_points) > 0:
                # 中心化
                center = np.mean(valid_points[:, :2], axis=0)
                points[:, :2] -= center
                
                # 自适应缩放
                scale = self._calculate_body_scale(points, n_points)
                if scale > 1e-6:
                    points[:, :2] /= scale
                    points[:, :2] = np.clip(points[:, :2], -1.0, 1.0)
            
            data.append(points)
        
        return np.array(data)
        
    def _calculate_body_scale(self, points, n_points):
        """计算身体尺度"""
        scale = 1.0
        if n_points >= 5:
            try:
                # 使用躯干关键点计算尺度
                left_shoulder = points[5] if n_points > 5 else points[1]
                right_shoulder = points[6] if n_points > 6 else points[2]
                left_hip = points[11] if n_points > 11 else points[3]
                right_hip = points[12] if n_points > 12 else points[4]
                
                if all(p[2] > 0.5 for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
                    shoulder_center = (left_shoulder[:2] + right_shoulder[:2]) / 2
                    hip_center = (left_hip[:2] + right_hip[:2]) / 2
                    scale = max(np.linalg.norm(shoulder_center - hip_center), 0.1)
            except:
                pass
        
        # 备用尺度方案
        if scale < 1e-6:
            valid_points = points[points[:, 2] >= self.min_confidence]
            if len(valid_points) > 0:
                scale = np.percentile(np.linalg.norm(valid_points[:, :2], axis=1), 75)
                scale = max(scale, 1e-6)
        
        return scale

    def _normalize_sequence(self, seq):
        """序列归一化"""
        if len(seq) < 5:
            return seq
        
        # 计算躯干尺度
        try:
            left_shoulder = seq[5] if len(seq) > 5 else seq[1]
            right_shoulder = seq[6] if len(seq) > 6 else seq[2]
            left_hip = seq[11] if len(seq) > 11 else seq[3]
            right_hip = seq[12] if len(seq) > 12 else seq[4]
            
            shoulder_center = (left_shoulder[:2] + right_shoulder[:2]) / 2
            hip_center = (left_hip[:2] + right_hip[:2]) / 2
            scale = np.linalg.norm(shoulder_center - hip_center)
            
            # 添加最小尺度保护
            scale = max(scale, 1e-6)
            normalized = seq.copy()
            normalized[:, :2] /= scale
            return normalized
        except:
            return seq
    
    def _calculate_distance(self, seq1, seq2):
        """计算两个序列之间的DTW距离"""
        seq1 = self._normalize_sequence(seq1)
        seq2 = self._normalize_sequence(seq2)
        
        if seq1.ndim == 1:
            seq1 = seq1.reshape(-1, 3)
        if seq2.ndim == 1:
            seq2 = seq2.reshape(-1, 3)
        
        def weighted_distance(x, y):
            if x[2] < self.min_confidence and y[2] < self.min_confidence:
                return 2.0 * self.oks_sigma
            
            # 使用几何平均权重
            weight = math.sqrt(x[2] * y[2]) if x[2] > 0 and y[2] > 0 else 0.1
            return euclidean(x[:2], y[:2]) * weight
        
        distance, _ = fastdtw(seq1, seq2, dist=weighted_distance)
        return distance
    
    def _calculate_oks(self, seq1, seq2, scale=1.0):
        """计算OKS (Object Keypoint Similarity) 分数"""
        k = len(seq1)
        if k != len(seq2):
            raise ValueError(f"关键点数量不一致: {k} vs {len(seq2)}")

        # 多尺度归一化
        scale_factors = []
        
        # 躯干尺度计算
        if len(seq1) >= 5:
            try:
                torso_points = [5, 6, 11, 12] if len(seq1) > 12 else [1, 2, 3, 4]
                shoulder_center = (seq1[torso_points[0]][:2] + seq1[torso_points[1]][:2]) / 2
                hip_center = (seq1[torso_points[2]][:2] + seq1[torso_points[3]][:2]) / 2
                torso_scale = np.linalg.norm(shoulder_center - hip_center)
                if torso_scale > 1e-6:
                    scale_factors.append(torso_scale)
            except:
                pass

        # 全局尺度
        valid_points = np.vstack([seq1[seq1[:,2] > 0], seq2[seq2[:,2] > 0]])
        if len(valid_points) > 0:
            global_scale = np.percentile(np.linalg.norm(valid_points[:,:2], axis=1), 75)
            if global_scale > 1e-6:
                scale_factors.append(global_scale)

        final_scale = np.median(scale_factors) if scale_factors else 1.0
        final_scale = max(final_scale, 1e-6)

        # 关键点匹配
        squared_distances = []
        weights = []
        valid_pair_counts = 0
        
        # 关键点类型标记
        point_types = np.zeros(k)
        if len(seq1) > 12:
            point_types[[5,6,11,12]] = 0  # 躯干
            point_types[[0,1,2,3,4,7,8,9,10,13,14]] = 1  # 四肢
        
        for i in range(k):
            if seq1[i,2] >= self.min_confidence or seq2[i,2] >= self.min_confidence:
                # 坐标补偿
                x1 = seq1[i,0] if seq1[i,2] >= self.min_confidence else seq2[i,0]
                y1 = seq1[i,1] if seq1[i,2] >= self.min_confidence else seq2[i,1]
                x2 = seq2[i,0] if seq2[i,2] >= self.min_confidence else seq1[i,0]
                y2 = seq2[i,1] if seq2[i,2] >= self.min_confidence else seq1[i,1]
                
                # 尺度归一化距离
                dx = (x1 - x2) / final_scale
                dy = (y1 - y2) / final_scale
                dist = math.sqrt(dx*dx + dy*dy)
                
                # 动态sigma
                sigma = self.oks_sigma * (1.5 if point_types[i] == 0 else 0.8)
                
                # 置信度权重
                conf_weight = min(seq1[i,2], seq2[i,2])
                weight = conf_weight * (1.2 if point_types[i] == 0 else 1.0)
                if conf_weight > 0.5:
                    weight *= 1.5
                
                # 距离截断
                clipped_dist = min(dist, 3*sigma)
                squared_distances.append(clipped_dist**2)
                weights.append(weight)
                valid_pair_counts += 1

        if valid_pair_counts == 0:
            return 0.0
        
        # 动态sigma调整
        mean_dist = np.mean(np.sqrt(squared_distances))
        dynamic_sigma = max(mean_dist * 0.8, self.oks_sigma)
        
        # 加权OKS计算
        var = 2 * dynamic_sigma**2
        exp_vals = np.exp(-np.array(squared_distances)/var)
        weighted_oks = np.sum(weights * exp_vals) / np.sum(weights)

        # 低分补偿
        if valid_pair_counts >= 0.5 * k and weighted_oks < 0.5:
            weighted_oks = min(0.5, weighted_oks * 2.0)
        
        return weighted_oks
    
    def match_action(self, input_csv, action_id):
        """
        匹配输入数据与指定动作的标准数据集
        """
        if action_id not in self.reference_data:
            print(f"未加载动作 {action_id} 的标准数据")
            return {
                'action_id': action_id,
                'match_ratio': 0.0,
                'avg_error': 1.0,
                'avg_oks': 0.0,
                'oks_score': 0.0,
                'dtw_score': 0.0,
                'combined_score': 0.0
            }
            
        # 获取该动作的匹配参数
        params = self.action_params.get(action_id, {})
        threshold = params.get("threshold", self.default_threshold)
        oks_sigma = params.get("oks_sigma", self.default_oks_sigma)
        min_confidence = params.get("min_confidence", self.default_min_confidence)
        
        # 临时设置参数
        old_threshold = self.threshold
        old_oks_sigma = self.oks_sigma
        old_min_confidence = self.min_confidence
        
        self.threshold = threshold
        self.oks_sigma = oks_sigma
        self.min_confidence = min_confidence
        
        try:
            # 加载并预处理输入数据
            if not os.path.exists(input_csv):
                print(f"输入文件不存在: {input_csv}")
                return self._get_zero_result(action_id)
                
            input_df = pd.read_csv(input_csv)
            input_data = self._preprocess_data(input_df)
            
            matched_count = 0
            oks_scores = []
            dtw_distances = []
            
            # 对输入数据中的每一帧进行匹配
            ref_data = self.reference_data[action_id]
            matched_indices = self.matched_indices[action_id]
            
            print(f"开始匹配动作 {action_id}，输入帧数: {len(input_data)}, 参考帧数: {len(ref_data)}")
            
            for input_seq in tqdm(input_data, desc=f"匹配动作 {action_id}"):
                min_distance = float('inf')
                max_oks = -1
                best_match_idx = -1
                
                # 在未匹配的标准数据中寻找最佳匹配
                for i, ref_seq in enumerate(ref_data):
                    if i in matched_indices:
                        continue
                    
                    try:
                        distance = self._calculate_distance(input_seq, ref_seq)
                        oks = self._calculate_oks(input_seq, ref_seq)
                        
                        if distance < self.threshold and distance < min_distance:
                            min_distance = distance
                            max_oks = oks
                            best_match_idx = i
                    except Exception as e:
                        print(f"匹配计算错误: {e}")
                        continue
                
                # 如果找到匹配
                if best_match_idx != -1:
                    matched_indices.add(best_match_idx)
                    self.matched_distances[action_id].append(min_distance)
                    oks_scores.append(max_oks)
                    dtw_distances.append(min_distance)
                    matched_count += 1
            
            # 计算评分
            total_reference = len(ref_data)
            match_ratio = matched_count / total_reference if total_reference > 0 else 0
            
            if dtw_distances:
                avg_distance = np.mean(dtw_distances)
                avg_error = avg_distance / self.threshold
            else:
                avg_distance = self.threshold
                avg_error = 1.0
            
            avg_oks = np.mean(oks_scores) if oks_scores else 0.0
            oks_score = min(10, avg_oks * 10)
            dtw_score = min(10, match_ratio * 10 * (1 - avg_error/3))
            combined_score = 0.7 * oks_score + 0.3 * dtw_score
            
            print(f"动作 {action_id} 匹配完成: 匹配率={match_ratio:.2%}, OKS={avg_oks:.3f}, 综合分数={combined_score:.2f}")
            
            return {
                'action_id': action_id,
                'match_ratio': match_ratio,
                'avg_error': avg_error,
                'avg_oks': avg_oks,
                'oks_score': oks_score,
                'dtw_score': dtw_score,
                'combined_score': combined_score
            }
            
        except Exception as e:
            print(f"匹配动作 {action_id} 时发生错误: {e}")
            return self._get_zero_result(action_id)
        finally:
            # 恢复原始参数
            self.threshold = old_threshold
            self.oks_sigma = old_oks_sigma
            self.min_confidence = old_min_confidence
    
    def _get_zero_result(self, action_id):
        """返回零分结果"""
        return {
            'action_id': action_id,
            'match_ratio': 0.0,
            'avg_error': 1.0,
            'avg_oks': 0.0,
            'oks_score': 0.0,
            'dtw_score': 0.0,
            'combined_score': 0.0
        }

    def match_multiple_actions(self, input_csv, action_ids):
        """批量匹配多个动作"""
        results = []
        for action_id in action_ids:
            if action_id in self.reference_data:
                result = self.match_action(input_csv, action_id)
                results.append(result)
            else:
                print(f"警告: 动作 {action_id} 的标准数据未加载，跳过")
                results.append(self._get_zero_result(action_id))
        return results

def main():
    """主函数，用于命令行测试"""
    parser = argparse.ArgumentParser(description='动作匹配评分系统')
    parser.add_argument('--reference', type=str, required=True, help='标准数据集CSV文件路径')
    parser.add_argument('--input', type=str, required=True, help='输入数据CSV文件路径')
    parser.add_argument('--threshold', type=float, default=0.5, help='匹配阈值')
    parser.add_argument('--oks_sigma', type=float, default=0.1, help='OKS算法的sigma参数')
    parser.add_argument('--min_confidence', type=float, default=0.15, help='最小置信度阈值')
    
    args = parser.parse_args()
    
    # 初始化匹配器
    matcher = ActionMatcher(
        reference_csvs={0: args.reference}, 
        threshold=args.threshold, 
        oks_sigma=args.oks_sigma, 
        min_confidence=args.min_confidence
    )
    
    # 进行匹配
    results = matcher.match_action(args.input, 0)
    
    # 输出结果
    print("\n匹配结果:")
    print(f"匹配比例: {results['match_ratio']:.2%}")
    print(f"平均误差: {results['avg_error']:.4f} (相对于阈值)")
    print(f"平均OKS分数: {results['avg_oks']:.4f}")
    print(f"OKS评分: {results['oks_score']:.2f}/10")
    print(f"DTW评分: {results['dtw_score']:.2f}/10")
    print(f"综合评分: {results['combined_score']:.2f}/10")

if __name__ == "__main__":
    main()