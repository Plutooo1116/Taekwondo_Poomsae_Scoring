## 姿态估计函数

import cv2
import numpy as np
import onnxruntime
import time
from collections import deque

class ONNXPoseEstimator:
    def __init__(self, onnx_path, conf_thres=0.25, iou_thres=0.45):
        """
        初始化ONNX姿态估计器
        
        参数:
            onnx_path: ONNX模型文件路径
            conf_thres: 置信度阈值
            iou_thres: IOU阈值
        """
        # 初始化ONNX运行时会话
        self.session = onnxruntime.InferenceSession(onnx_path, 
                                                  providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # 模型参数
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        
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

    def preprocess(self, image):
        """
        预处理输入图像
        
        参数:
            image: 输入图像 (BGR格式)
            
        返回:
            blob: 预处理后的blob
            original_shape: 原始图像尺寸
        """
        # 记录原始尺寸
        original_shape = image.shape[:2]
        
        # 调整大小并归一化
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (self.input_shape[3], self.input_shape[2]), 
                                    swapRB=True, crop=False)
        
        return blob, original_shape

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

    def postprocess(self, outputs, original_shape):
        """
        后处理模型输出
        
        参数:
            outputs: 模型输出
            original_shape: 原始图像尺寸
            
        返回:
            results: 处理后的结果 (包含边界框和关键点)
        """
        # 转置和重塑输出
        outputs = np.transpose(outputs, (0, 2, 1))
        outputs = outputs.reshape(1, -1, outputs.shape[2])
        
        # 提取预测
        predictions = np.squeeze(outputs, 0)
        
        # 过滤低置信度的预测
        mask = predictions[:, 4] > self.conf_threshold
        predictions = predictions[mask]

         # 将YOLO格式的边界框(center_x, center_y, width, height)转换为(x1, y1, x2, y2)
        boxes = predictions[:, :4].copy()
        boxes[:, 0] = predictions[:, 0] - predictions[:, 2] / 2  # x1 = center_x - width/2
        boxes[:, 1] = predictions[:, 1] - predictions[:, 3] / 2  # y1 = center_y - height/2
        boxes[:, 2] = predictions[:, 0] + predictions[:, 2] / 2  # x2 = center_x + width/2
        boxes[:, 3] = predictions[:, 1] + predictions[:, 3] / 2  # y2 = center_y + height/2
        
        # 扩展界面边界框 (增加10%的尺寸)
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        boxes[:, 0] -= width * 0.1  # 向左扩展5%
        boxes[:, 1] -= height * 0.1  # 向上扩展5%
        boxes[:, 2] += width * 0.1   # 向右扩展5%
        boxes[:, 3] += height * 0.1  # 向下扩展5%

        scores = predictions[:, 4]
        indices = self.non_max_suppression(boxes, scores, self.iou_threshold)
        predictions = predictions[indices]
        boxes = boxes[indices]
        
        # 提取关键点 (假设关键点信息在预测的后17*3个值中)
        keypoints = predictions[:, 5:5+17*3].reshape(-1, 17, 3)
        
        # 缩放回原始图像尺寸
        scale_h = original_shape[0] / self.input_shape[2]
        scale_w = original_shape[1] / self.input_shape[3]
        
        # 缩放边界框
        boxes[:, [0, 2]] *= scale_w
        boxes[:, [1, 3]] *= scale_h

        # 缩放关键点
        keypoints[:, :, 0] *= scale_w
        keypoints[:, :, 1] *= scale_h
        
        results = []
        for i in range(len(predictions)):
            box = boxes[i]
            score = predictions[i, 4]
            kpts = keypoints[i]
            
            # 转换17点→15点
            transformed_kpts = self.transform_keypoints(kpts)
            
            # 平滑处理
            smoothed_kpts = self.smoother.smooth_keypoints(transformed_kpts)
            
            results.append({'box': box, 'score': score, 'keypoints': smoothed_kpts})
            
        return results

    def non_max_suppression(self, boxes, scores, threshold):
        """
        非极大值抑制
        
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
            
        返回:
            results: 检测结果
            vis_image: 可视化后的图像
            inference_time: 推理时间(毫秒)
        """
        # 预处理
        start_time = time.time()
        blob, original_shape = self.preprocess(image)
        
        # 推理
        outputs = self.session.run([self.output_name], {self.input_name: blob})[0]
        
        # 后处理
        results = self.postprocess(outputs, original_shape)
        inference_time = (time.time() - start_time) * 1000  # 毫秒
        # 确保inference_time不会为零或负值
        inference_time = max(inference_time, 0.001)  # 最小设置为0.001毫秒

        # 可视化
        vis_image = self.visualize(image, results, draw_skeleton)
        
        return results, vis_image, inference_time


class KeypointSmoother:
    """关键点平滑滤波器"""
    def __init__(self, window_size=5, min_confidence=0.3):
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.history = deque(maxlen=window_size)
        
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
        total_weight = 0.0
        
        # 加权移动平均（最近帧权重更高）
        for i, kpts in enumerate(reversed(self.history)):
            weight = 1.0 / (i + 1)  # 指数衰减权重
            for j in range(15):
                if kpts[j, 2] > self.min_confidence:
                    smoothed[j] += kpts[j] * weight
            total_weight += weight
        
        # 归一化
        if total_weight > 0:
            smoothed /= total_weight
        
        # 处理低置信度关键点
        for j in range(15):
            if current_keypoints[j, 2] < self.min_confidence:
                valid_history = [kpts[j] for kpts in self.history if kpts[j, 2] > self.min_confidence]
                if valid_history:
                    avg_point = np.mean(valid_history, axis=0)
                    smoothed[j] = avg_point
        
        return smoothed