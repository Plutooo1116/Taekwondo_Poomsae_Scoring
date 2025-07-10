## 姿态估计函数 - RKNN版本（最新版）

import cv2
import numpy as np
from rknnlite.api import RKNNLite
import time
from math import exp
from collections import deque

class RKNNPoseEstimator:
    def __init__(self, rknn_path, conf_thres=0.25, iou_thres=0.45):
        """
        初始化RKNN姿态估计器
        
        参数:
            rknn_path: RKNN模型文件路径
            conf_thres: 置信度阈值
            iou_thres: IOU阈值
        """
        # 初始化RKNN运行时会话
        self.session = RKNNLite()
        ret = self.session.load_rknn(rknn_path)
        if ret != 0:
            raise Exception(f"加载RKNN模型失败: {rknn_path}")
        
        ret = self.session.init_runtime()
        if ret != 0:
            raise Exception("初始化RKNN运行时环境失败")
        
        # 启用详细日志定位瓶颈
        self.session.load_rknn(rknn_path)

        # 多核
        ret = self.session.init_runtime(
            core_mask=RKNNLite.NPU_CORE_0_1_2
        )

        # 模型参数
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        
        # RKNN模型配置
        self.input_shape = [1, 3, 640, 640]  # NCHW格式
        self.class_num = 1
        self.headNum = 3
        self.keypoint_num = 17
        self.strides = [8, 16, 32]
        self.mapSize = [[80, 80], [40, 40], [20, 20]]
        self.objectThresh = conf_thres
        self.nmsThresh = iou_thres
        
        # 生成网格点
        self.meshgrid = []
        self._generate_meshgrid()
        
        # 关键点映射配置 (COCO17 → ST-GCN coco_cut14)
        self.COC017_TO_COCO_CUT = {
            0: 0,   # 鼻子 → 鼻子
            5: 1,   # 左肩 → 左肩
            6: 2,   # 右肩 → 右肩
            7: 3,   # 左肘 → 左肘
            8: 4,   # 右肘 → 右肘
            9: 5,   # 左手腕 → 左手腕
            10: 6,  # 右手腕 → 右手腕
            11: 7,  # 左髋 → 左髋
            12: 8,  # 右髋 → 右髋
            13: 9,  # 左膝 → 左膝
            14: 10, # 右膝 → 右膝
            15: 11, # 左脚踝 → 左脚踝
            16: 12  # 右脚踝 → 右脚踝
        }
        
        # 右侧关节索引 (需要镜像)
        self.RIGHT_JOINT_INDICES = [2, 4, 6, 8, 10, 12]  # coco_cut中的右肩、右肘等
        
        # 15个关键点的连接关系
        self.skeleton = [
            ((0, 14), 'head'),  # 鼻子-脖子
            ((1, 14), 'head'),  # 左肩-脖子
            ((2, 14), 'head'),  # 右肩-脖子
            ((1, 3), 'arms'),   # 左肩-左肘
            ((2, 4), 'arms'),   # 右肩-右肘
            ((3, 5), 'arms'),   # 左肘-左手腕
            ((4, 6), 'arms'),   # 右肘-右手腕
            ((1, 7), 'body'),   # 左肩-左髋
            ((2, 8), 'body'),   # 右肩-右髋
            ((7, 9), 'legs'),   # 左髋-左膝
            ((8, 10), 'legs'),  # 右髋-右膝
            ((9, 11), 'legs'),  # 左膝-左脚踝
            ((10, 12), 'legs'), # 右膝-右脚踝
            ((7, 13), 'body'),  # 左髋-骨盆
            ((8, 13), 'body'),  # 右髋-骨盆
            ((7, 8), 'body')    # 左髋-右髋
        ]
        
        # 颜色定义
        self.colors = {
            'head': (255, 0, 0),    # 蓝色
            'body': (0, 255, 0),    # 绿色
            'arms': (255, 165, 0),  # 橙色
            'legs': (255, 0, 255)   # 紫色
        }
        
        # 关键点名称 (COCO格式)
        self.keypoint_names = {
            0: "nose",
            1: "left_shoulder",
            2: "right_shoulder",
            3: "left_elbow",
            4: "right_elbow",
            5: "left_wrist",
            6: "right_wrist",
            7: "left_hip",
            8: "right_hip",
            9: "left_knee",
            10: "right_knee",
            11: "left_ankle",
            12: "right_ankle",
            13: "pelvis",
            14: "neck"
        }

        # 初始化关键点平滑器
        self.smoother = KeypointSmoother(window_size=5, min_confidence=0.3)

    def _generate_meshgrid(self):
        """生成网格点坐标"""
        self.meshgrid = []
        for index in range(self.headNum):
            for i in range(self.mapSize[index][0]):
                for j in range(self.mapSize[index][1]):
                    self.meshgrid.append(j + 0.5)
                    self.meshgrid.append(i + 0.5)

    def preprocess(self, image):
        """
        预处理输入图像
        
        参数:
            image: 输入图像 (BGR格式)
            
        返回:
            input_data: 预处理后的数据
            original_shape: 原始图像尺寸
        """
        # 记录原始尺寸
        original_shape = image.shape[:2]
        
        # 调整图像大小
        resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
        # BGR转RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # 添加batch维度
        input_data = np.expand_dims(rgb_image, 0)
        
        return input_data, original_shape

    def transform_keypoints(self, keypoints):
        """转换17个关键点到15个关键点格式"""
        # 1. 创建15个关键点的空数组
        transformed = np.zeros((15, 3))
        
        # 2. 映射保留的13个关节
        for orig_idx, new_idx in self.COC017_TO_COCO_CUT.items():
            transformed[new_idx] = keypoints[orig_idx]
        
        # 3. 计算骨盆中心（左髋11 + 右髋12）
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        pelvis = (left_hip + right_hip) / 2
        pelvis_conf = min(left_hip[2], right_hip[2])
        
        # 4. 计算脖子中心（左肩5 + 右肩6）
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        neck = (left_shoulder + right_shoulder) / 2
        neck_conf = min(left_shoulder[2], right_shoulder[2])
        
        # 5. 设置骨盆和脖子点
        transformed[13] = [pelvis[0], pelvis[1], pelvis_conf]  # 骨盆
        transformed[14] = [neck[0], neck[1], neck_conf]       # 脖子
        
        return transformed

    def _sigmoid(self, x):
        """Sigmoid激活函数"""
        return 1 / (1 + exp(-x))

    def _calculate_iou(self, box1, box2):
        """计算两个边界框的交并比"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def _apply_nms(self, detections):
        """应用非极大值抑制"""
        if len(detections) == 0:
            return []
        
        # 按置信度排序
        detections.sort(key=lambda x: x['score'], reverse=True)
        
        keep = []
        for i, det in enumerate(detections):
            if det['score'] < self.conf_threshold:
                continue
                
            keep_detection = True
            for kept_det in keep:
                if self._calculate_iou(det['box'], kept_det['box']) > self.iou_threshold:
                    keep_detection = False
                    break
            
            if keep_detection:
                keep.append(det)
        
        return keep

    def postprocess(self, outputs, original_shape):
        """
        后处理模型输出
        
        参数:
            outputs: 模型输出
            original_shape: 原始图像尺寸
            
        返回:
            results: 处理后的结果 (包含边界框和关键点)
        """
        detections = []
        
        # 重新整理输出数据
        output = []
        for i in range(len(outputs)):
            output.append(outputs[i].reshape((-1)))
        
        scale_h = original_shape[0] / 640
        scale_w = original_shape[1] / 640
        
        gridIndex = -2
        
        for index in range(self.headNum):
            reg = output[index * 2 + 0]      # 回归输出
            cls = output[index * 2 + 1]      # 分类输出
            pose = output[self.headNum * 2 + index]  # 姿态输出
            
            for h in range(self.mapSize[index][0]):
                for w in range(self.mapSize[index][1]):
                    gridIndex += 2
                    
                    # 计算分类置信度
                    if self.class_num == 1:
                        cls_max = self._sigmoid(cls[0 * self.mapSize[index][0] * self.mapSize[index][1] + h * self.mapSize[index][1] + w])
                        cls_index = 0
                    else:
                        cls_max = 0
                        cls_index = 0
                        for cl in range(self.class_num):
                            cls_val = cls[cl * self.mapSize[index][0] * self.mapSize[index][1] + h * self.mapSize[index][1] + w]
                            if cl == 0 or cls_val > cls_max:
                                cls_max = cls_val
                                cls_index = cl
                        cls_max = self._sigmoid(cls_max)
                    
                    if cls_max > self.objectThresh:
                        # DFL解码边界框
                        regdfl = []
                        for lc in range(4):
                            sfsum = 0
                            locval = 0
                            
                            # 计算softmax
                            for df in range(16):
                                temp = exp(reg[((lc * 16) + df) * self.mapSize[index][0] * self.mapSize[index][1] + h * self.mapSize[index][1] + w])
                                reg[((lc * 16) + df) * self.mapSize[index][0] * self.mapSize[index][1] + h * self.mapSize[index][1] + w] = temp
                                sfsum += temp
                            
                            # 计算期望值
                            for df in range(16):
                                sfval = reg[((lc * 16) + df) * self.mapSize[index][0] * self.mapSize[index][1] + h * self.mapSize[index][1] + w] / sfsum
                                locval += sfval * df
                            regdfl.append(locval)
                        
                        # 计算边界框坐标（恢复原始正确的计算方式）
                        x1 = (self.meshgrid[gridIndex + 0] - regdfl[0]) * self.strides[index]
                        y1 = (self.meshgrid[gridIndex + 1] - regdfl[1]) * self.strides[index]
                        x2 = (self.meshgrid[gridIndex + 0] + regdfl[2]) * self.strides[index]
                        y2 = (self.meshgrid[gridIndex + 1] + regdfl[3]) * self.strides[index]
                        
                        # 转换到原图坐标系
                        xmin = max(0, min(x1 * scale_w, original_shape[1]))
                        ymin = max(0, min(y1 * scale_h, original_shape[0]))
                        xmax = max(0, min(x2 * scale_w, original_shape[1]))
                        ymax = max(0, min(y2 * scale_h, original_shape[0]))
                        
                        # 解析关键点
                        keypoints = []
                        for kc in range(self.keypoint_num):
                            px = pose[(kc * 3 + 0) * self.mapSize[index][0] * self.mapSize[index][1] + h * self.mapSize[index][1] + w]
                            py = pose[(kc * 3 + 1) * self.mapSize[index][0] * self.mapSize[index][1] + h * self.mapSize[index][1] + w]
                            vs = self._sigmoid(pose[(kc * 3 + 2) * self.mapSize[index][0] * self.mapSize[index][1] + h * self.mapSize[index][1] + w])
                            
                            x = (px * 2.0 + (self.meshgrid[gridIndex + 0] - 0.5)) * self.strides[index] * scale_w
                            y = (py * 2.0 + (self.meshgrid[gridIndex + 1] - 0.5)) * self.strides[index] * scale_h
                            
                            keypoints.append([x, y, vs])
                        
                        detection = {
                            'box': [xmin, ymin, xmax, ymax],
                            'score': cls_max,
                            'keypoints': np.array(keypoints)
                        }
                        detections.append(detection)
        
        # 应用NMS
        filtered_detections = self._apply_nms(detections)
        
        # 处理每个检测结果
        results = []
        for detection in filtered_detections:
            box = detection['box']
            
            # 在这里应用边界框扩展（增加10%的尺寸）
            width = box[2] - box[0]
            height = box[3] - box[1]
            expand_w = width * 0.05  # 左右各扩展5%
            expand_h = height * 0.05  # 上下各扩展5%
            
            expanded_box = [
                max(0, box[0] - expand_w),
                max(0, box[1] - expand_h), 
                min(original_shape[1], box[2] + expand_w),
                min(original_shape[0], box[3] + expand_h)
            ]
            
            # 转换17点→15点
            transformed_kpts = self.transform_keypoints(detection['keypoints'])
            
            # 平滑处理
            smoothed_kpts = self.smoother.smooth_keypoints(transformed_kpts)
            
            results.append({
                'box': expanded_box, 
                'score': detection['score'], 
                'keypoints': smoothed_kpts
            })
        
        return results

    def non_max_suppression(self, boxes, scores, threshold):
        """
        非极大值抑制（保持与原接口兼容）
        
        参数:
            boxes: 边界框数组
            scores: 分数数组
            threshold: IOU阈值
            
        返回:
            keep: 保留的索引
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

        return keep

    def visualize(self, image, results, draw_skeleton=True):
        """
        可视化结果
        
        参数:
            image: 原始图像 (BGR格式)
            results: 检测结果
            draw_skeleton: 是否绘制骨架
            
        返回:
            vis_image: 可视化后的图像
        """
        vis_image = image.copy()
    
        for result in results:
            box = result['box']
            score = result['score']
            keypoints = result['keypoints']
            
            # 如果draw_skeleton为True，绘制骨架连接
            if draw_skeleton:
                # 绘制边界框
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制分数
                label = f"Person: {score:.2f}"
                cv2.putText(vis_image, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 绘制关键点
                for i, kpt in enumerate(keypoints):
                    x, y, conf = kpt
                    if conf > 0.3:  # 只绘制置信度高的关键点
                        cv2.circle(vis_image, (int(x), int(y)), 5, (0, 255, 0), -1)
                        # 显示关键点编号
                        cv2.putText(vis_image, str(i), (int(x)+5, int(y)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

                drawn_connections = set()
                for (connection, body_part) in self.skeleton:
                    # 标准化连接方向 (确保(0,1)和(1,0)被视为相同)
                    sorted_conn = tuple(sorted(connection))
                    
                    # 如果已经绘制过这个连接，跳过
                    if sorted_conn in drawn_connections:
                        continue
                    
                    # 检查连接索引是否有效
                    if (connection[0] >= len(keypoints)) or (connection[1] >= len(keypoints)):
                        continue
                        
                    pt1, pt2 = keypoints[connection[0]], keypoints[connection[1]]
                    
                    # 检查关键点数据是否完整
                    if len(pt1) >= 3 and len(pt2) >= 3:
                        if pt1[2] > 0.3 and pt2[2] > 0.3:  # 只绘制置信度高的连接
                            cv2.line(vis_image, 
                                    (int(pt1[0]), int(pt1[1])), 
                                    (int(pt2[0]), int(pt2[1])), 
                                    self.colors[body_part], 2)
                            drawn_connections.add(sorted_conn)
        
        return vis_image

    def predict(self, image, draw_skeleton=True):
        """
        执行预测
        
        参数:
            image: 输入图像 (BGR格式)
            draw_skeleton: 是否绘制骨架
            
        返回:
            results: 检测结果
            vis_image: 可视化后的图像
            inference_time: 推理时间(毫秒)
        """
        # 预处理
        start_time = time.time()
        input_data, original_shape = self.preprocess(image)
        
        # 推理
        outputs = self.session.inference(inputs=[input_data])
        
        # 后处理
        results = self.postprocess(outputs, original_shape)
        inference_time = (time.time() - start_time) * 1000  # 毫秒
        
        # 可视化
        vis_image = self.visualize(image, results, draw_skeleton)
        
        return results, vis_image, inference_time

    def release(self):
        """释放资源"""
        if hasattr(self, 'session'):
            self.session.release()

    def __del__(self):
        """析构函数，确保资源被释放"""
        self.release()


class KeypointSmoother:
    """关键点平滑滤波器"""
    def __init__(self, window_size=5, min_confidence=0.3):
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.history = deque(maxlen=window_size)
        
        # 针对不同身体部位设置不同的平滑参数
        self.body_part_configs = {
            'head': {'smooth_factor': 0.7, 'min_conf': 0.5},      # 头部平滑度中等
            'torso': {'smooth_factor': 0.8, 'min_conf': 0.4},     # 躯干平滑度高
            'arms': {'smooth_factor': 0.6, 'min_conf': 0.3},      # 手臂平滑度中等
            'legs': {'smooth_factor': 0.9, 'min_conf': 0.2}       # 腿部平滑度最高
        }
        
        # 关键点分组
        self.keypoint_groups = {
            'head': [0],                    # 鼻子
            'torso': [1, 2, 7, 8, 13, 14], # 肩膀、髋部、骨盆、脖子
            'arms': [3, 4, 5, 6],          # 手臂关键点
            'legs': [9, 10, 11, 12]        # 腿部关键点（包括脚踝）
        }
        
    def _get_keypoint_group(self, keypoint_idx):
        """获取关键点所属的身体部位组"""
        for group_name, indices in self.keypoint_groups.items():
            if keypoint_idx in indices:
                return group_name
        return 'torso'  # 默认分组
        
    def _adaptive_smooth(self, current_kpt, history_kpts, group_config):
        """自适应平滑算法"""
        if len(history_kpts) < 2:
            return current_kpt
            
        # 计算历史点的稳定性
        stability_scores = []
        for i in range(1, len(history_kpts)):
            prev_kpt = history_kpts[i-1]
            curr_hist_kpt = history_kpts[i]
            
            # 计算位置变化
            pos_diff = np.sqrt((curr_hist_kpt[0] - prev_kpt[0])**2 + 
                             (curr_hist_kpt[1] - prev_kpt[1])**2)
            
            # 计算置信度变化
            conf_diff = abs(curr_hist_kpt[2] - prev_kpt[2])
            
            # 综合稳定性分数（位置变化小且置信度变化小则稳定性高）
            stability = 1.0 / (1.0 + pos_diff * 0.1 + conf_diff * 2.0)
            stability_scores.append(stability)
        
        avg_stability = np.mean(stability_scores) if stability_scores else 0.5
        
        # 根据稳定性调整平滑因子
        smooth_factor = group_config['smooth_factor']
        if avg_stability < 0.3:  # 不稳定时增加平滑
            smooth_factor = min(0.95, smooth_factor + 0.2)
        elif avg_stability > 0.8:  # 稳定时减少平滑
            smooth_factor = max(0.3, smooth_factor - 0.1)
            
        # 加权平均
        if current_kpt[2] > group_config['min_conf']:
            # 当前点置信度高，使用当前点和历史点的加权平均
            valid_history = [kpt for kpt in history_kpts if kpt[2] > group_config['min_conf']]
            if valid_history:
                hist_avg = np.mean(valid_history, axis=0)
                smoothed = smooth_factor * hist_avg + (1 - smooth_factor) * current_kpt
            else:
                smoothed = current_kpt
        else:
            # 当前点置信度低，更多依赖历史点
            valid_history = [kpt for kpt in history_kpts if kpt[2] > group_config['min_conf']]
            if valid_history:
                smoothed = np.mean(valid_history, axis=0)
                smoothed[2] = max(current_kpt[2], smoothed[2] * 0.8)  # 适当降低置信度
            else:
                smoothed = current_kpt
                
        return smoothed
        
    def smooth_keypoints(self, current_keypoints):
        """应用时域平滑滤波"""
        if len(current_keypoints) != 15:
            return current_keypoints
        
        # 将当前帧加入历史
        self.history.append(current_keypoints.copy())
        
        # 如果历史数据不足，直接返回当前帧
        if len(self.history) < 2:
            return current_keypoints
        
        smoothed = np.zeros_like(current_keypoints)
        
        # 对每个关键点进行自适应平滑
        for i in range(15):
            group_name = self._get_keypoint_group(i)
            group_config = self.body_part_configs[group_name]
            
            # 获取该关键点的历史数据（除了最后一帧，即当前帧）
            history_kpts = []
            for frame_idx in range(len(self.history) - 1):
                history_kpts.append(self.history[frame_idx][i])
            
            current_kpt = current_keypoints[i]
            
            # 应用自适应平滑
            smoothed[i] = self._adaptive_smooth(current_kpt, history_kpts, group_config)
            
        return smoothed