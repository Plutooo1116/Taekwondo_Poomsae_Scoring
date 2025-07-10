## 查看最近历史视频

import cv2
import os
from PyQt5.QtWidgets import (QWidget, QLabel, QMessageBox, QDesktopWidget, 
                            QVBoxLayout, QHBoxLayout, QPushButton)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap,QIcon, QPainter

class SimpleVideoPlayer(QWidget):
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.is_valid = False
        
        if not self._validate_video_file(video_path):
            return
        
        try:
            self.setWindowIcon(QIcon("./9.10 1523/assets/Taekwondo.png"))
        except:
            pass
        
        self.is_valid = True
        self.cap = cv2.VideoCapture(video_path)
        
        self._init_ui()
        self._init_video_info()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(self.frame_delay)
    
    def _validate_video_file(self, video_path):
        if not os.path.exists(video_path):
            return False
            
        if not os.access(video_path, os.R_OK):
            return False
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        else:
            cap.release()
            return True
    
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
            painter.fillRect(self.rect(), Qt.white)

    def _init_ui(self):
        """初始化UI元素"""
        self.setWindowTitle(f"视频回放: {os.path.basename(self.video_path)}")
        
        # 主布局 - 垂直布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        
        # 设置视频区域的最小尺寸
        screen = QDesktopWidget().screenGeometry()
        self.video_label.setMinimumSize(int(screen.width()*0.5), int(screen.height()*0.6))
        
        main_layout.addWidget(self.video_label, 1)  # 视频区域占主要空间
        
        # 按钮区域 - 水平布局
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 10, 0, 0)
        # 添加弹簧使按钮居中
        button_layout.addStretch(1)  # 左侧弹簧

        # 重播按钮
        self.btn_replay = QPushButton("重播")
        self.btn_replay.setFixedSize(160, 40)
        self.btn_replay.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.9);
                color: black;
                border: 2px solid rgb(160, 199, 255);
                border-radius: 10px;
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
        button_layout.addWidget(self.btn_replay)
        
        # 添加按钮间距
        button_layout.addSpacing(50)

        # 返回按钮
        self.btn_return = QPushButton("返回")
        self.btn_return.setFixedSize(160, 40)
        self.btn_return.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.9);
                color: black;
                border: 2px solid rgb(160, 199, 255);
                border-radius: 10px;
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
        button_layout.addWidget(self.btn_return)
        
        button_layout.addStretch(1)  # 右侧弹簧
    
        # 将按钮布局直接添加到主布局
        main_layout.addLayout(button_layout)
        
        # 设置窗口初始大小
        self.resize(int(screen.width()*0.6), int(screen.height()*0.7))
    
    def _init_video_info(self):
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_delay = int(2000 / self.fps) if self.fps > 0 else 24
        
        self.setWindowTitle(f"视频回放: {os.path.basename(self.video_path)} ({self.fps:.1f} FPS, {self.frame_count} 帧)")
    
    def _update_frame(self):
        if not self.is_valid:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            # 视频播放结束，停止定时器
            self.timer.stop()
            return
        
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.width(), 
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    def replay_video(self):
        """重播视频"""
        if self.is_valid and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if not self.timer.isActive():
                self.timer.start(self.frame_delay)
    
    def closeEvent(self, event):
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
            
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
            
        super().closeEvent(event)
    
    def show(self):
        if self.is_valid:
            # 连接按钮信号
            self.btn_return.clicked.connect(self.close)
            self.btn_replay.clicked.connect(self.replay_video)
            super().show()
        else:
            error_msg = f"无法打开视频文件: {self.video_path}"
            
            if not os.path.exists(self.video_path):
                error_msg += "\n原因: 文件不存在"
            elif not os.access(self.video_path, os.R_OK):
                error_msg += "\n原因: 没有读取权限"
            else:
                error_msg += "\n原因: 可能是文件损坏或格式不支持"
            
            QMessageBox.critical(None, "视频播放错误", error_msg)