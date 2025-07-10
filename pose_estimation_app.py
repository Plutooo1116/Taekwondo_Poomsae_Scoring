## 开始训练界面——骨骼——分数——保存

import cv2
import numpy as np
import time
import os
import csv
import math
from datetime import datetime
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, 
                             QFileDialog, QMessageBox, QFrame, QScrollArea,QDesktopWidget,QSizePolicy, QComboBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QColor
from onnx_pose_estimator import ONNXPoseEstimator
import pandas as pd
import numpy as np
from stgcn_visualizer_onnx import init_onnx_model, process_window_onnx as process_window
from fast_dtw import ActionMatcher
from action_config import (STAGE_EVALUATION_PARAMS, STGCN_CLASS_NAMES, 
                          get_stage_params, get_matching_stages)
# from rknn_pose_estimator import RKNNPoseEstimator
from history_window import HistoryWindow

class PoseEstimationApp(QMainWindow):
    return_signal = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 初始化ST-GCN模型
        print("正在初始化ST-GCN模型...")
        self.stgcn_model = init_onnx_model()  # Assuming init_onnx_model is the correct function to initialize the ST-GCN model
        print("ST-GCN模型初始化完成")
        
        # 使用配置文件中的动作评分参数
        self.action_params = STAGE_EVALUATION_PARAMS
        
        # 初始化动作匹配器
        self.action_matcher = ActionMatcher()
        
        self.setWindowTitle("跆拳道品势训练")
        # 获取屏幕尺寸并设置窗口大小
        screen = QDesktopWidget().screenGeometry()
        self.setMinimumSize(int(screen.width()*0.7), int(screen.height()*0.7))

        try:
            self.setWindowIcon(QIcon("./9.10 1523/assets/Taekwondo.png"))
        except:
            pass
        
        # 定义动作阶段
        self.stages = [
            "品势准备",
            "1. 左转下格挡",
            "2. 前行步冲拳",
            "3. 右转下格挡",
            "4. 前行步冲拳",
            "5. 转向前弓步下格挡",
            "6. 向前冲拳",
            "7. 右转前行步中格挡",
            "8. 前行步冲拳",
            "9. 左转前行步中格挡",
            "10. 前行步冲拳",
            "11. 弓步下格挡",
            "12. 向前冲拳",
            "13. 左转前行步上格挡",
            "14. 前踢前行步冲拳",
            "15. 转身前行步上格挡",
            "16. 前踢前行步冲拳",
            "17. 向后转弓步下格挡",
            "18. 弓步冲拳",
            "结束动作"
        ]
        self.stage_scores = [0.0] * len(self.stages)  # 初始化各阶段分数

        self.MIN_KEYPOINTS = 15  # 最少需要检测到的关键点数量

        self.show_skeleton = True  # 默认显示骨骼

        self.unsaved_changes = False  

        # 控制是否开始录制
        self.recording_started = False

        # CSV记录相关变量
        self.csv_writer = None
        self.csv_file = None
        self.frame_count = 0
        self.csv_path = None  # 保存CSV文件路径用于后续分析

        # Initialize history window
        self.history_window = HistoryWindow()

        # 创建倒计时标签
        self.countdown_label = QLabel()
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet("""
            font-size: 160px; 
            font-weight: bold; 
            color: white;
            background-color: rgba(0, 0, 0, 150);
            border-radius: 50px;
            padding: 20px;
            min-width: 150px;
        """)

        # 初始化倒计时定时器
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_seconds = 5

        # 初始化主界面
        self.init_ui()
        
        # 创建一个覆盖层容器用于居中显示倒计时
        self.overlay_widget = QWidget(self.video_label)
        self.overlay_widget.setLayout(QVBoxLayout())
        self.overlay_widget.layout().setContentsMargins(0, 0, 0, 0)
        self.overlay_widget.layout().addWidget(self.countdown_label, 0, Qt.AlignCenter)
        self.overlay_widget.setVisible(False)
        self.overlay_widget.setStyleSheet("background-color: transparent;")
        self.overlay_widget.setAttribute(Qt.WA_TransparentForMouseEvents)  # 允许鼠标事件穿透

        # 初始化姿态估计器
        self.estimator = ONNXPoseEstimator("yolo11n-pose.onnx")
        # self.estimator = RKNNPoseEstimator("./model/yolo11-pose.rknn")
        
        # 初始化视频捕获
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
        # 评分相关变量
        self.pose_scores = {}
        self.total_score = 0.0
        self.completed_poses = 0
     
    def paintEvent(self, event):
        """重绘事件，绘制背景图片"""
        painter = QPainter(self)
        
        # 绘制背景图片
        try:
            # 加载背景图片
            bg_pixmap = QPixmap("./9.10 1523/assets/background.png")
            if not bg_pixmap.isNull():
                # 缩放图片以完全填充窗口
                scaled_pixmap = bg_pixmap.scaled(
                    self.size(), 
                    Qt.IgnoreAspectRatio,  # 忽略宽高比，完全填充
                    Qt.SmoothTransformation
                )
                # 绘制背景图片
                painter.drawPixmap(0, 0, scaled_pixmap)
        except Exception as e:
            print(f"背景图片绘制错误: {e}")
            # 如果图片加载失败，使用纯色背景
            painter.fillRect(self.rect(), QColor(248, 245, 255))
        
    def init_ui(self):
        # 主窗口部件
        central_widget = QWidget()
        central_widget.setStyleSheet("background: transparent;")  # 设置中央部件透明
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 左侧视频显示区域
        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(0, 0, 0, 0)

        # 返回按钮
        self.btn_return = QPushButton("返回")
        self.btn_return.clicked.connect(self.on_return_clicked)
        left_panel.addWidget(self.btn_return, alignment=Qt.AlignLeft | Qt.AlignTop)  # 靠左上对齐
        self.btn_return.setStyleSheet("""
            QPushButton {
                min-width: 100px;  /* 增加宽度 */
                max-width: 100px;
                background-color: rgba(255, 255, 255, 0.9);
                color: black;
                border: 2px solid rgb(160, 199, 255);
                border-radius: 10px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: rgba(240, 240, 240, 0.9);
                color: rgb(140, 179, 235);
                border-color: rgb(140, 179, 235);
            }
            QPushButton:pressed {
                background-color: rgba(230, 230, 230, 0.9);
                color: rgb(120, 159, 215);
                border-color: rgb(120, 159, 215);
            }
        """)

        # 视频显示标签 
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(720, 640)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 添加尺寸策略
        self.video_label.setScaledContents(False)
        left_panel.addWidget(self.video_label, stretch=1)  # 添加stretch参数
        
        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        # 下拉框
        self.training_mode_combo = QComboBox()
        self.training_mode_combo.addItems([
            "品势一章",
            "品势准备",
            "转身下格挡",
            "前行步冲拳",
            "转向前弓步下格挡冲拳",
            "转身前行步中格挡",
            "弓步下格挡冲拳",
            "转身前行步上格挡",
            "前踢前行步冲拳",
            "向后转弓步下格挡",
            "弓步冲拳（哈）",
            "结束动作"
        ])
        # 设置下拉框样式
        self.training_mode_combo.setStyleSheet("""
            QComboBox {
                min-width: 150px;
                height: 32px;
                background-color: rgba(255, 255, 255, 0.9);
                color: black;
                border: 2px solid rgb(160, 199, 255);
                border-radius: 10px;
                padding: 5px 10px;
                font-size: 16px;
            }
            QComboBox:hover {
                border-color: rgb(140, 179, 235);
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: rgb(160, 199, 255);
                border-left-style: solid;
                border-top-right-radius: 10px;
                border-bottom-right-radius: 10px;
            }
            QComboBox::down-arrow {
                image: url(arrow_down.png);
                width: 10px;
                height: 16px;
            }
            QComboBox QAbstractItemView {
                border: 2px solid rgb(160, 199, 255);
                border-radius: 10px;
                background-color: rgba(255, 255, 255, 0.9);
                selection-background-color: rgb(200, 230, 255);
                selection-color: black;
                padding: 5px;
            }
        """)

        button_layout.addWidget(self.training_mode_combo)
        


        # 开始训练按钮
        self.btn_start = QPushButton("开始训练")
        self.btn_start.clicked.connect(self.start_training)
        button_layout.addWidget(self.btn_start)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.9);
                color: black;
                border: 2px solid rgb(160, 199, 255);
                border-radius: 10px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: rgba(240, 240, 240, 0.9);
                color: rgb(140, 179, 235);
                border-color: rgb(140, 179, 235);
            }
            QPushButton:pressed {
                background-color: rgba(230, 230, 230, 0.9);
                color: rgb(120, 159, 215);
                border-color: rgb(120, 159, 215);
            }
        """)
        
        # 加载视频按钮
        self.btn_load = QPushButton("加载视频")
        self.btn_load.clicked.connect(self.load_media)
        button_layout.addWidget(self.btn_load)
        self.btn_load.setStyleSheet("""
            QPushButton {
               background-color: rgba(255, 255, 255, 0.9);
                color: black;
                border: 2px solid rgb(160, 199, 255);
                border-radius: 10px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: rgba(240, 240, 240, 0.9);
                color: rgb(140, 179, 235);
                border-color: rgb(140, 179, 235);
            }
            QPushButton:pressed {
                background-color: rgba(230, 230, 230, 0.9);
                color: rgb(120, 159, 215);
                border-color: rgb(120, 159, 215);
            }
        """)
        
        # 暂停/继续按钮
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.clicked.connect(self.pause_video)
        button_layout.addWidget(self.btn_pause)
        self.btn_pause.setStyleSheet("""
            QPushButton {
               background-color: rgba(255, 255, 255, 0.9);
                color: black;
                border: 2px solid rgb(160, 199, 255);
                border-radius: 10px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: rgba(240, 240, 240, 0.9);
                color: rgb(140, 179, 235);
                border-color: rgb(140, 179, 235);
            }
            QPushButton:pressed {
                background-color: rgba(230, 230, 230, 0.9);
                color: rgb(120, 159, 215);
                border-color: rgb(120, 159, 215);
            }
        """)

        # 匹配骨骼按钮
        self.btn_skeleton = QPushButton("脱离骨骼")  # 初始显示"脱离骨骼"
        self.btn_skeleton.setCheckable(True)
        self.btn_skeleton.setChecked(True)  # 默认选中状态(显示骨骼)
        self.btn_skeleton.clicked.connect(self.toggle_skeleton)
        button_layout.addWidget(self.btn_skeleton)
        self.btn_skeleton.setStyleSheet("""
            QPushButton {
               background-color: rgba(255, 255, 255, 0.9);
                color: black;
                border: 2px solid rgb(160, 199, 255);
                border-radius: 10px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: rgba(240, 240, 240, 0.9);
                color: rgb(140, 179, 235);
                border-color: rgb(140, 179, 235);
            }
            QPushButton:pressed {
                background-color: rgba(230, 230, 230, 0.9);
                color: rgb(120, 159, 215);
                border-color: rgb(120, 159, 215);
            }
        """)

        # 在按钮区域添加保存按钮
        self.btn_save = QPushButton("保存")
        self.btn_save.clicked.connect(self.save_scores)
        button_layout.addWidget(self.btn_save) 
        self.btn_save.setStyleSheet("""
            QPushButton {
               background-color: rgba(255, 255, 255, 0.9);
                color: black;
                border: 2px solid rgb(160, 199, 255);
                border-radius: 10px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: rgba(240, 240, 240, 0.9);
                color: rgb(140, 179, 235);
                border-color: rgb(140, 179, 235);
            }
            QPushButton:pressed {
                background-color: rgba(230, 230, 230, 0.9);
                color: rgb(120, 159, 215);
                border-color: rgb(120, 159, 215);
            }
        """)
        
        left_panel.addLayout(button_layout)
        
        # 右侧评分区域
        right_panel = QWidget()  # 创建一个QWidget作为容器
        right_panel.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.2);  /* 白色背景,20%不透明度 */
        """)
        right_layout = QVBoxLayout(right_panel)  # 为right_panel设置布局

        # 总分显示（初始显示）
        self.total_score_label = QLabel("总分: 0")
        self.total_score_label.setAlignment(Qt.AlignCenter)
        self.total_score_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        right_layout.addWidget(self.total_score_label)

        # 添加分隔线
        self.score_divider = QFrame()
        self.score_divider.setFrameShape(QFrame.HLine)
        self.score_divider.setFrameShadow(QFrame.Sunken)
        right_layout.addWidget(self.score_divider)
        
        # 阶段评分标题
        stage_label = QLabel("阶段评分")
        stage_label.setAlignment(Qt.AlignCenter)
        stage_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        right_layout.addWidget(stage_label)
        
        # 阶段评分滚动区域
        scroll_area = QScrollArea()
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QWidget#scrollContent {
                background-color: transparent;
            }
        """)
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # 创建阶段评分标签
        self.stage_labels = []
        for i, stage in enumerate(self.stages):
            label = QLabel(f"{stage}: 0")
            label.setStyleSheet("font-size: 20px; margin: 3px;")
            self.stage_labels.append(label)
            scroll_layout.addWidget(label)
        
        # 添加伸缩项使内容顶部对齐
        scroll_layout.addStretch()
        
        # 设置滚动区域内容
        scroll_area.setWidget(scroll_content)
        right_layout.addWidget(scroll_area)

        # 保存scroll_area为实例变量以便后续访问
        self.scroll_area = scroll_area
        self.scroll_content = scroll_content
        
        # 将左右面板添加到主布局
        main_layout.addLayout(left_panel, 70)  # 左侧使用QVBoxLayout
        main_layout.addWidget(right_panel, 30)  # 右侧使用QWidget

    def analyze_pose_sequence(self, csv_path):
        """使用ST-GCN和FastDTW分析动作序列并自动评分"""
        try:
            print("开始分析动作序列...")
            
            # 1. 使用ST-GCN进行动作分类
            print("正在使用ST-GCN进行动作分类...")
            frames = self.load_csv_data(csv_path)
            window_predictions = self.classify_with_stgcn(frames)
            
            # 2. 对每个分类窗口进行标准动作匹配
            print("正在与标准动作匹配...")
            stage_evaluation_count = {}  # 记录每个阶段的评估次数
            stage_scores_sum = {}  # 记录每个阶段的分数总和
            
            for win in window_predictions:
                predicted_class = win['predicted_class']
                action_name = STGCN_CLASS_NAMES.get(predicted_class, "未知动作")
                
                # 查找对应的阶段名称列表
                stage_names = get_matching_stages(action_name)
                if not stage_names:
                    print(f"未找到匹配的阶段: {action_name}")
                    continue
                
                # 对于每个可能匹配的阶段，进行评估
                for stage_name in stage_names:
                    # 获取该阶段的参数
                    params = get_stage_params(stage_name)
                    reference_csv = params.get("reference_csv")
                    
                    if not reference_csv or not os.path.exists(reference_csv):
                        print(f"标准数据文件不存在: {reference_csv}")
                        continue
                    
                    # 初始化匹配器
                    matcher = ActionMatcher(
                        reference_csvs={predicted_class: reference_csv},
                        threshold=params.get("threshold", 3.0),
                        oks_sigma=params.get("oks_sigma", 0.2),
                        min_confidence=params.get("min_confidence", 0.2)
                    )
                    
                    # 进行匹配
                    results = matcher.match_action(csv_path, predicted_class)
                    
                    # 累计分数
                    score = min(results['combined_score'] / 10.0, 1.0)  # 转换为0-1范围
                    weight = params.get("weight", 1.0)  # 获取权重
                    weighted_score = score * weight
                    
                    # 更新累计分数
                    if stage_name not in stage_evaluation_count:
                        stage_evaluation_count[stage_name] = 0
                        stage_scores_sum[stage_name] = 0.0
                    
                    stage_evaluation_count[stage_name] += 1
                    stage_scores_sum[stage_name] += weighted_score
                    
                    print(f"阶段: {stage_name}, 动作: {action_name}, 分数: {score:.3f}, 权重: {weight}")
            
            # 3. 计算加权平均分数并更新UI
            for stage_name in stage_evaluation_count:
                if stage_evaluation_count[stage_name] > 0:
                    avg_score = stage_scores_sum[stage_name] / stage_evaluation_count[stage_name]
                    stage_index = self.get_stage_index(stage_name)
                    if stage_index >= 0:
                        self.onPoseEvaluated(stage_index, avg_score, f"ST-GCN+DTW评估")
                        print(f"最终阶段评分: {stage_name} = {avg_score:.3f}")
            
            print("动作分析完成!")
            return True
            
        except Exception as e:
            print(f"动作分析错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def find_matching_stage(self, action_name):
        """根据动作名称查找匹配的阶段"""
        # 动作名称到阶段的映射
        action_to_stage = {
            "preparation": "品势准备",
            "forward_stance_low_block": "左转下格挡",  # 或右转下格挡，根据具体情况
            "forward_stance_punch": "前行步冲拳",
            "front_punch": "向前冲拳",
            "low_block": "弓步下格挡",
            "middle_block": "右转前行步中格挡",  # 或左转，根据具体情况
            "rising_block": "左转前行步上格挡",
            "front_kick_punch": "前踢前行步冲拳",
            "turn_forward_stance_low_block": "向后转弓步下格挡",
            "turn_forward_stance_punch": "弓步冲拳"
        }
        
        # 首先尝试直接映射
        if action_name in action_to_stage:
            return action_to_stage[action_name]
        
        # 如果没有直接映射，尝试模糊匹配
        for key, stage in action_to_stage.items():
            if key in action_name or action_name in key:
                return stage
        
        return None
    
    def get_stage_index(self, stage_name):
        """根据阶段名称获取索引"""
        # 简化的阶段名称，去除数字前缀
        clean_stage_name = stage_name.replace("左转", "").replace("右转", "").replace("转身", "").replace("向前", "").replace("向后转", "").strip()
        
        for i, stage in enumerate(self.stages):
            # 去除阶段序号进行匹配
            clean_stage = stage.split('.', 1)[-1].strip() if '.' in stage else stage.strip()
            
            if (stage_name in stage or 
                clean_stage_name in clean_stage or 
                stage.replace(".", "").strip() == stage_name or
                any(keyword in stage for keyword in stage_name.split()) or
                any(keyword in stage_name for keyword in clean_stage.split())):
                return i
        
        # 如果没有精确匹配，尝试关键词匹配
        for i, stage in enumerate(self.stages):
            if "准备" in stage_name and "准备" in stage:
                return i
            elif "下格挡" in stage_name and "下格挡" in stage:
                return i
            elif "冲拳" in stage_name and "冲拳" in stage:
                return i
            elif "中格挡" in stage_name and "中格挡" in stage:
                return i
            elif "上格挡" in stage_name and "上格挡" in stage:
                return i
            elif "前踢" in stage_name and "前踢" in stage:
                return i
            elif "结束" in stage_name and "结束" in stage:
                return i
        
        print(f"警告: 未找到阶段 '{stage_name}' 对应的索引")
        return -1
    
    def load_csv_data(self, csv_path):
        """加载CSV骨架数据"""
        df = pd.read_csv(csv_path)
        frames = []
        for _, row in df.iterrows():
            frame_num = int(row['frame_num'])
            kps = row[1:].values.reshape(15, 3)  # 转换为(15,3)数组: x,y,conf
            frames.append({
                'frame_num': frame_num,
                'keypoints': kps
            })
        return frames
    
    def classify_with_stgcn(self, frames):
        """使用ST-GCN模型进行分类"""
        # 准备数据
        coords = np.array([f['keypoints'][:, :2] for f in frames])
        confs = np.array([f['keypoints'][:, 2] for f in frames])
        motions = np.zeros_like(coords)
        motions[1:] = coords[1:] - coords[:-1]
        
        # 合并为3通道 (x, y, conf)
        coords_3ch = np.stack([coords[:, :, 0], coords[:, :, 1], confs], axis=2)
        
        # 处理每个窗口
        window_size = 20
        window_stride = 10
        window_predictions = []
        
        for start_idx in range(0, len(frames) - window_size + 1, window_stride):
            end_idx = start_idx + window_size
            coord_window = coords_3ch[start_idx:end_idx].transpose(1, 2, 0)
            motion_window = motions[start_idx:end_idx].transpose(1, 2, 0)
            
            pred_class, probs = process_window(coord_window, motion_window, self.stgcn_model)
            
            window_predictions.append({
                'start_frame': start_idx,
                'end_frame': end_idx - 1,
                'predicted_class': pred_class,
                'probabilities': probs
            })
        
        return window_predictions
    
    def finish_recording_and_analyze(self):
        """结束录制并开始分析"""
        if self.csv_path and os.path.exists(self.csv_path):
            print(f"开始分析文件: {self.csv_path}")
            success = self.analyze_pose_sequence(self.csv_path)
            if success:
                QMessageBox.information(self, "分析完成", "动作分析完成！请查看评分结果。")
            else:
                QMessageBox.warning(self, "分析失败", "动作分析过程中出现错误。")

    def toggle_skeleton(self):
        """切换是否显示骨骼"""
        self.show_skeleton = not self.show_skeleton  # 切换显示状态
        
        # 更新按钮文本
        if self.show_skeleton:
            self.btn_skeleton.setText("脱离骨骼")
        else:
            self.btn_skeleton.setText("匹配骨骼")
        
        # 立即更新当前帧
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.process_frame(frame)
    
    def onPoseEvaluated(self, pose, score, feedback):
        """更新阶段评分"""
        # 确保pose在有效范围内
        if 0 <= pose < len(self.stages):
            # 更新分数
            self.stage_scores[pose] = score
            
            # 标记有未保存的修改
            self.unsaved_changes = True

            # 更新显示
            percent = f"{score*100:.1f}"
            if score > 0.8:
                color = "color: green;"
            elif score > 0.5:
                color = "color: orange;"
            else:
                color = "color: red;"
            
            self.stage_labels[pose].setText(f"{self.stages[pose]}: {percent}")
            self.stage_labels[pose].setStyleSheet(f"font-size: 16px; margin: 3px; {color}")

            # 更新总分
            self.total_score = np.mean([s for s in self.stage_scores if s > 0])
            self.total_score_label.setText(f"总分: {self.total_score*100:.1f}")

        # 标记有未保存的修改
        self.unsaved_changes = True

        # 更新总分
        self.update_total_score()

    
    def update_total_score(self):
        """更新总分显示"""
        # 设置颜色
        if self.total_score > 0.8:
            color = "color: green;"
        elif self.total_score > 0.5:
            color = "color: orange;"
        else:
            color = "color: red;"
        
        # 如果是单个动作模式，更新当前分数显示
        if hasattr(self, 'current_score_label') and self.current_score_label:
            self.current_score_label.setText(f"当前分数: {self.total_score*100:.1f}")
            self.current_score_label.setStyleSheet(f"font-size: 24px; color: blue; margin-bottom: 10px; {color}")
    
    def on_return_clicked(self):
        """返回按钮点击事件处理"""
        # 检查是否有未保存的评分数据
        if hasattr(self, 'unsaved_changes') and self.unsaved_changes:
            # 如果有未保存的修改，显示确认对话框
            reply = QMessageBox.question(
                self,
                '未保存的修改',
                '您有未保存的评分数据，确定要返回吗?',
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            
            if reply == QMessageBox.Save:
                self.save_scores()
                # 等待保存完成
                time.sleep(0.5)  # 短暂延迟确保保存完成
                self.return_signal.emit()
                self.close()
            elif reply == QMessageBox.Discard:
                self.return_signal.emit()
                self.close()
            # 如果选择Cancel，什么都不做
        else:
            # 没有未保存的修改，直接返回
            self.return_signal.emit()
            self.close()  
        
    def start_training(self):
        # 停止当前训练并释放资源
        self.stop_video()

        selected_mode = self.training_mode_combo.currentText()
        print(f"开始训练模式: {selected_mode}")  # 调试输出
        
        # 根据选择的模式调整训练逻辑
        if selected_mode == "品势一章":
            # 完整品势一章训练
            self.stages = [
                "品势准备",
                "1. 左转下格挡",
                "2. 前行步冲拳",
                "3. 右转下格挡",
                "4. 前行步冲拳",
                "5. 转向前弓步下格挡",
                "6. 向前冲拳",
                "7. 右转前行步中格挡",
                "8. 前行步冲拳",
                "9. 左转前行步中格挡",
                "10. 前行步冲拳",
                "11. 弓步下格挡",
                "12. 向前冲拳",
                "13. 左转前行步上格挡",
                "14. 前踢前行步冲拳",
                "15. 转身前行步上格挡",
                "16. 前踢前行步冲拳",
                "17. 向后转弓步下格挡",
                "18. 弓步冲拳",
                "结束动作"
            ]
            self.show_full_score_panel()  # 显示完整评分面板
        else:
            # 单个动作训练
            self.stages = [selected_mode]
            self.show_single_movement_panel(selected_mode)  # 显示单个动作评分面板
        
        # 重置分数
        self.stage_scores = [0.0] * len(self.stages)
        self.update_total_score()
        
        # 清除历史评分
        self.pose_scores = {}
        self.total_score = 0.0
        self.completed_poses = 0
        self.update_total_score()
        
        # 重置录制标志
        self.recording_started = False

        # 禁用按钮防止重复点击
        self.btn_start.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.btn_pause.setEnabled(False)
        
        # 重置未保存标志
        self.unsaved_changes = False

        # 开始实际训练
        self.actual_start_training()

    def show_full_score_panel(self):
        """显示完整的评分面板（品势一章模式）"""
        # 创建新的内容widget
        new_content = QWidget()
        self.scroll_layout = QVBoxLayout(new_content)
        
        # 确保总分和分隔线可见
        self.total_score_label.setVisible(True)
        self.score_divider.setVisible(True)
        
        # 重新创建阶段评分标签
        self.stage_labels = []
        for i, stage in enumerate(self.stages):
            label = QLabel(f"{stage}: 0")
            label.setStyleSheet("font-size: 20px; margin: 3px;")
            self.stage_labels.append(label)
            self.scroll_layout.addWidget(label)
        
        # 添加伸缩项使内容顶部对齐
        self.scroll_layout.addStretch()
        
        # 更新滚动区域内容
        self.scroll_area.setWidget(new_content)
        self.scroll_content = new_content

    def show_single_movement_panel(self, movement):
        """显示单个动作评分面板"""
        # 创建新的内容widget
        new_content = QWidget()
        self.scroll_layout = QVBoxLayout(new_content)
        
        # 隐藏总分和分隔线
        self.total_score_label.setVisible(False)
        self.score_divider.setVisible(False)

        # 添加动作名称标题
        title_label = QLabel(movement)
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; margin-bottom: 15px;")
        self.scroll_layout.addWidget(title_label)
        
        # 添加当前分数显示（不显示总分）
        self.current_score_label = QLabel("当前分数: 0")
        self.current_score_label.setStyleSheet("font-size: 20px; color: blue; margin-bottom: 15px;")
        self.scroll_layout.addWidget(self.current_score_label)
        
        # 添加动作要点说明
        tips_label = QLabel(self.get_movement_tips(movement))
        tips_label.setStyleSheet("font-size: 22px; margin-top: 15px;")
        tips_label.setWordWrap(True)
        self.scroll_layout.addWidget(tips_label)
        
        # 添加伸缩项使内容顶部对齐
        self.scroll_layout.addStretch()
        
        # 更新滚动区域内容
        self.scroll_area.setWidget(new_content)
        self.scroll_content = new_content

    def get_movement_tips(self, movement):
        """获取动作要点说明"""
        tips = {
            "品势准备": "双脚开立形成并排步，双手握拳置于腹前形成准备姿势。",
            "转身下格挡": "左转为例: 左脚向前跟步,身体向左旋转90°,同时左下格挡。",
            "前行步冲拳": "右脚为例：右脚上步形成右走步，右手直拳。",
            "转向前弓步下格挡冲拳": "左脚向左移步,带动身体向左旋转90°形成左弓步,同时,做左下格挡,而后右手直拳",
            "转身前行步中格挡": "右转为例：右脚向前跟步,身体向右旋转90°,同时左中内格挡",
            "弓步下格挡冲拳": "右脚向右移步,带动身体向右旋转90°形成右弓步,同时,做右下格挡,而后左手直拳",
            "转身前行步上格挡": "左转为例：左脚向左移步,带动身体向左旋转90°形成左弓步,同时左上格挡",
            "前踢前行步冲拳": "右脚为例：右脚前踢，前落地形成右走步,同时右手直拳。",
            "向后转弓步下格挡": "左脚向右移步,带动身体向右旋转90°形成左弓步,同时左手下格挡。",
            "弓步冲拳": "右脚向前移形成右弓步,右手直拳并配合发声。",
            "结束动作": "以右前脚掌为轴,带动身体向后旋转180°,左脚回收形成还原姿势"
        }
        return tips.get(movement, "暂无具体动作要点说明")

    def update_countdown(self):
        self.countdown_seconds -= 1
        self.countdown_label.setText(str(self.countdown_seconds))
        
        if self.countdown_seconds <= 0:
            # 倒计时结束
            self.countdown_timer.stop()
            self.overlay_widget.setVisible(False)
            
            # 设置开始录制标志
            self.recording_started = True

            # 强制刷新显示
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.process_frame(frame)

            # 启用按钮
            self.btn_start.setEnabled(True)
            self.btn_load.setEnabled(True)
            self.btn_pause.setEnabled(True)
    
    def actual_start_training(self):
        """实际开始训练的逻辑"""
        # 初始化视频录制
        video_dir = "training_videos"
        os.makedirs(video_dir, exist_ok=True)
        # 使用时间戳作为文件名，避免冲突
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(video_dir, f"training_{timestamp}.mp4")
        
        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (640, 480)  # 根据您的摄像头分辨率调整
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 30.0, frame_size)
            
        # 在单独的线程中尝试打开摄像头
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Windows上使用DSHOW
        # self.cap = cv2.VideoCapture('/dev/video21', cv2.CAP_V4L2)
        
        # 检查摄像头是否打开
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", "无法打开摄像头")
            # 启用按钮
            self.btn_start.setEnabled(True)
            self.btn_load.setEnabled(True)
            self.btn_pause.setEnabled(True)
            return
        
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 显示倒计时标签 - 只在摄像头成功打开后显示
        self.countdown_seconds = 5
        self.countdown_label.setText(str(self.countdown_seconds))
        
        # 调整覆盖层大小和位置
        self.overlay_widget.setParent(self.video_label)
        self.overlay_widget.setGeometry(0, 0, self.video_label.width(), self.video_label.height())
        self.overlay_widget.setVisible(True)
        self.countdown_label.show()
        
        # 启动倒计时定时器
        self.countdown_timer.start(1000)  # 1秒间隔
        
        # 启动视频帧定时器 (但先不处理帧，直到倒计时结束)
        self.timer.start(33)  # ~30fps

        # 初始化CSV记录
        self.init_csv_recording()
    
    def init_csv_recording(self):
        """初始化CSV记录文件"""
        # 创建记录目录
        os.makedirs("pose_records", exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join("pose_records", f"pose_{timestamp}.csv")
        
        # 打开CSV文件
        self.csv_file = open(self.csv_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # 写入CSV头部
        header = ['frame_num']
        for i in range(15):  # 15个关键点
            header.extend([f'kp{i}_x', f'kp{i}_y', f'kp{i}_conf'])
        self.csv_writer.writerow(header)
        
        self.frame_count = 0

    def close_csv_recording(self):
        """关闭CSV记录文件"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def record_keypoints_to_csv(self, keypoints):
        """将关键点数据记录到CSV文件"""
        if not self.csv_writer or not self.recording_started:
            return
        
        # 准备数据行
        row = [self.frame_count]
        for kp in keypoints:
            row.extend([kp[0], kp[1], kp[2]])  # x, y, confidence
        
        # 写入CSV
        self.csv_writer.writerow(row)
        self.frame_count += 1

    def load_media(self):
        # 停止当前播放
        self.stop_video()
        
        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开媒体文件", "", 
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if not file_path:
            return
        
        # 初始化CSV记录
        self.init_csv_recording()
        self.recording_started = True  # 立即开始记录

        # 根据文件类型处理
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            self.process_image(file_path)
        else:
            self.process_video(file_path)
    
    def process_image(self, file_path):
        # 读取图片
        image = cv2.imread(file_path)
        if image is None:
            QMessageBox.warning(self, "错误", "无法加载图片文件")
            return
        
        # 处理图片
        self.process_frame(image)
    
    def process_video(self, file_path):
        # 打开视频文件
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", "无法打开视频文件")
            return
        
        # 初始化视频写入器（如果需要保存处理后的视频）
        video_dir = "processed_videos"
        os.makedirs(video_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(video_dir, f"processed_{timestamp}.mp4")
        
        # 获取视频帧尺寸和FPS
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, fps, (frame_width, frame_height))
        
        # 启动定时器
        self.timer.start(33)
    
    def pause_video(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_pause.setText("继续")
        else:
            self.timer.start(33)
            self.btn_pause.setText("暂停")
    
    def stop_video(self):
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        # 确保视频写入器已关闭
        if hasattr(self, 'video_writer') and self.video_writer and self.video_writer.isOpened():
            self.video_writer.release()
        
        # 清空视频显示
        self.video_label.clear()
        self.btn_pause.setText("暂停")

        # 关闭CSV记录并开始分析
        if self.csv_file:
            self.close_csv_recording()
            # 在视频结束后自动开始分析
            if self.csv_path:
                self.finish_recording_and_analyze()

    def save_scores(self):
        # 检查是否有有效的评分数据可保存
        if self.total_score <= 0 and all(score <= 0 for score in self.stage_scores):
            QMessageBox.warning(self, "警告", "当前没有可保存的评分数据！")
            return
        
        # 获取当前日期时间
        current_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # 准备保存数据
        try:
            # 准备视频文件信息 - 更安全的处理方式
            video_file = None
            if hasattr(self, 'video_path') and self.video_path:
                try:
                    # 确保视频写入器完成写入
                    if hasattr(self, 'video_writer') and self.video_writer and self.video_writer.isOpened():
                        self.video_writer.release()
                    video_file = os.path.basename(self.video_path)
                except Exception as e:
                    print(f"视频文件处理错误: {e}")
                    video_file = None

            # 准备CSV文件信息 - 更安全的处理方式
            csv_file = None
            if hasattr(self, 'csv_path') and self.csv_path:
                try:
                    if self.csv_file:
                        self.csv_file.close()  # 关闭CSV文件
                        self.csv_file = None
                        self.csv_writer = None
                    csv_file = os.path.basename(self.csv_path)
                except Exception as e:
                    print(f"CSV文件处理错误: {e}")
                    csv_file = None
            
            # 构建要保存的数据 - 确保所有值都是字符串
            saved_data = {
                "date": current_datetime,
                "total_score": f"{self.total_score*100:.1f}" if self.total_score >= 0 else "0.0",
                "stage_scores": [
                    f"{stage}: {score*100:.1f}" if score >= 0 else f"{stage}: 0.0"
                    for stage, score in zip(self.stages, self.stage_scores) 
                ],
                "video_file": video_file if video_file else "无视频文件",
                "csv_file": csv_file if csv_file else "无数据文件"
            }
            
            # 初始化历史记录窗口
            if not hasattr(self, 'history_window') or not self.history_window:
                self.history_window = HistoryWindow()
            
            # 添加记录到历史窗口
            self.history_window.add_record(
                saved_data["date"],
                saved_data["total_score"],
                saved_data["stage_scores"],
                saved_data["video_file"],
                saved_data["csv_file"]
            )
            
            # 保存成功后重置未保存标志
            self.unsaved_changes = False
            QMessageBox.information(self, "成功", "评分已保存到历史记录！")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存过程中发生错误: {str(e)}")
        finally:
            # 确保资源被释放
            if hasattr(self, 'video_writer') and self.video_writer and self.video_writer.isOpened():
                try:
                    self.video_writer.release()
                except:
                    pass
            if hasattr(self, 'csv_file') and self.csv_file:
                try:
                    self.csv_file.close()
                except:
                    pass
    
    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.timer.stop()
            return
        
        ret, frame = self.cap.read()
        if not ret:
            # 视频结束，停止并分析
            self.stop_video()
            return
        
        # 执行姿态估计并获取带骨骼点的帧
        results, vis_frame, _ = self.estimator.predict(frame, draw_skeleton=self.show_skeleton)
        
        # 写入带骨骼点的视频帧
        if hasattr(self, 'video_writer') and self.video_writer and self.video_writer.isOpened() and self.recording_started:
            self.video_writer.write(vis_frame) 

        self.process_frame(frame)
    
    def process_frame(self, frame):
        # 执行姿态估计
        results, vis_frame, inference_time = self.estimator.predict(frame, draw_skeleton=self.show_skeleton)
        
        # 显示FPS
        if inference_time > 0 and not math.isinf(inference_time):
            fps = 1000.0 / inference_time
        else:
            fps = 0
        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 检测人体和关键点
        if not results or len(results) == 0:  # 没有检测到人体
            # 仅在初始检测时显示提示
            if not hasattr(self, 'full_body_detected'):
                cv2.putText(vis_frame, "Waiting for person detection...", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            try:
                person = results[0]  # 获取第一个检测到的人体
                keypoints = person['keypoints']  # 直接从字典获取关键点
                
                # 检查是否检测到完整人体
                valid_kpts = [kp for kp in keypoints if kp[2] > 0.1]
                if len(valid_kpts) >= self.MIN_KEYPOINTS:  # 检测到完整人体
                    self.full_body_detected = True
                else:  # 检测到不完整人体
                    if not hasattr(self, 'full_body_detected'):
                        cv2.putText(vis_frame, "Partial body detected", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # 记录关键点到CSV
                if self.recording_started:
                    self.record_keypoints_to_csv(keypoints)

                # 设置固定的小窗口大小
                small_h = 200
                small_w = 150
                
                # 创建纯骨骼图 - 黑色背景上只绘制骨骼
                skeleton_only = np.zeros((small_h, small_w, 3), dtype=np.uint8)
                
                # 获取骨骼连接信息
                skeleton_edges = [(conn[0][0], conn[0][1]) for conn in self.estimator.skeleton]
                
                # 计算关键点边界框
                valid_kpts = [kp for kp in keypoints if kp[2] > 0.1]
                if valid_kpts:
                    x_coords = [kp[0] for kp in valid_kpts]
                    y_coords = [kp[1] for kp in valid_kpts]
                    min_x, max_x = min(x_coords), max(x_coords)
                    min_y, max_y = min(y_coords), max(y_coords)
                    
                    # 计算缩放比例和偏移量
                    scale_x = small_w / (max_x - min_x) if (max_x - min_x) > 0 else 1
                    scale_y = small_h / (max_y - min_y) if (max_y - min_y) > 0 else 1
                    scale = min(scale_x, scale_y) * 0.8  # 稍微缩小一点，留出边距
                    
                    offset_x = (small_w - (max_x - min_x) * scale) / 2
                    offset_y = (small_h - (max_y - min_y) * scale) / 2
                    
                    # 绘制骨骼连接线
                    for start_idx, end_idx in skeleton_edges:
                        if (start_idx < len(keypoints) and end_idx < len(keypoints) and 
                        keypoints[start_idx][2] > 0.1 and keypoints[end_idx][2] > 0.1):
                            # 计算缩放后的坐标
                            start_x = int((keypoints[start_idx][0] - min_x) * scale + offset_x)
                            start_y = int((keypoints[start_idx][1] - min_y) * scale + offset_y)
                            end_x = int((keypoints[end_idx][0] - min_x) * scale + offset_x)
                            end_y = int((keypoints[end_idx][1] - min_y) * scale + offset_y)
                            
                            # 确保坐标在小窗口范围内
                            start_x = max(0, min(small_w-1, start_x))
                            start_y = max(0, min(small_h-1, start_y))
                            end_x = max(0, min(small_w-1, end_x))
                            end_y = max(0, min(small_h-1, end_y))
                            
                            cv2.line(skeleton_only, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                    
                    # 绘制关键点
                    for kp in keypoints:
                        if kp[2] > 0.1:  # 只绘制置信度高的关键点
                            x = int((kp[0] - min_x) * scale + offset_x)
                            y = int((kp[1] - min_y) * scale + offset_y)
                            x = max(0, min(small_w-1, x))
                            y = max(0, min(small_h-1, y))
                            cv2.circle(skeleton_only, (x, y), 3, (0, 0, 255), -1)
                
                # 将小窗口放在右下角
                y_start = vis_frame.shape[0] - small_h - 10
                x_start = vis_frame.shape[1] - small_w - 10
                
                # 添加半透明背景和白色边框
                overlay = vis_frame.copy()
                cv2.rectangle(overlay, (x_start-5, y_start-5), 
                            (x_start+small_w+5, y_start+small_h+5), 
                            (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, vis_frame, 0.5, 0, vis_frame)
                cv2.rectangle(vis_frame, (x_start-2, y_start-2), 
                            (x_start+small_w+2, y_start+small_h+2), 
                            (255, 255, 255), 1)
                
                # 放置骨骼图
                vis_frame[y_start:y_start+small_h, x_start:x_start+small_w] = skeleton_only
        
            except Exception as e:
                print(f"Keypoint processing error: {e}")
                if not hasattr(self, 'full_body_detected'):
                    cv2.putText(vis_frame, "Detection error", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 转换为Qt图像格式并显示
        self.display_image(vis_frame)
        
    
    def display_image(self, image):
        # 将OpenCV图像转换为Qt图像
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 缩放图像以适应标签大小
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # 创建新的QPixmap作为背景，确保居中
        background = QPixmap(self.video_label.size())
        background.fill(Qt.black)  # 黑色背景
        painter = QPainter(background)
        painter.drawPixmap(
            (self.video_label.width() - scaled_pixmap.width()) // 2,
            (self.video_label.height() - scaled_pixmap.height()) // 2,
            scaled_pixmap
        )
        painter.end()
        
        self.video_label.setPixmap(background)
        self.video_label.setAlignment(Qt.AlignCenter)

        # 更新覆盖层位置和大小
        if self.overlay_widget.isVisible():
            self.overlay_widget.setGeometry(0, 0, self.video_label.width(), self.video_label.height())
    
    
    def closeEvent(self, event):
        self.stop_video()
        super().closeEvent(event)