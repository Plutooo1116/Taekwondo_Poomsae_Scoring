from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QSpacerItem, QSizePolicy, QDesktopWidget, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFont, QColor, QPixmap, QImage, QPainter, QBrush
from pose_estimation_app import PoseEstimationApp
from history_window import HistoryWindow

class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("初始界面")
        self.setWindowFlags(Qt.FramelessWindowHint)  # 隐藏原生标题栏
        
        # 获取屏幕尺寸并设置窗口大小
        screen = QDesktopWidget().screenGeometry()
        self.setMinimumSize(int(screen.width()*0.3), int(screen.height()*0.4))
        
        # 创建自定义标题栏
        self.title_bar = QWidget()
        self.title_bar.setFixedHeight(40)
        self.title_bar.setStyleSheet("""
            background: qlineargradient(
                x1:0, y1:0,
                x2:1, y2:0,
                stop:0 rgb(241, 243, 255),
                stop:1 rgb(241, 243, 255)
            );
        """)
        
        # 标题文字
        self.title_label = QLabel("跆拳道品势训练系统")
        self.title_label.setStyleSheet("""
            QLabel {
                color: black;
                font-size: 16px;
                font-weight: bold;
                padding-left: 15px;
            }
        """)
        
        # 关闭按钮
        self.close_btn = QPushButton("×")
        self.close_btn.setFixedSize(40, 40)
        self.close_btn.setStyleSheet("""
            QPushButton {
                color: black;
                font-size: 20px;
                border: none;
                background: transparent;
            }
            QPushButton:hover {
                background-color: rgb(160, 199, 255);
            }
        """)
        self.close_btn.clicked.connect(self.close)
        
        # 标题栏布局
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.close_btn)
        title_layout.setContentsMargins(0, 0, 5, 0)
        
        # 初始化UI
        self.init_ui()
    
    
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
        
        painter.end()
    
    def extract_black_parts(self, image_path):
        """提取图片中的黑色部分"""
        try:
            image = QImage(image_path)
            # 创建一个透明背景的图像
            result = QImage(image.size(), QImage.Format_ARGB32)
            result.fill(Qt.transparent)
            
            # 遍历所有像素，提取黑色部分
            for x in range(image.width()):
                for y in range(image.height()):
                    color = image.pixelColor(x, y)
                    # 判断是否为黑色或接近黑色
                    if color.black() > 200:  # 黑色阈值
                        result.setPixelColor(x, y, QColor(0, 0, 0, 255))  # 纯黑色
                    else:
                        result.setPixelColor(x, y, Qt.transparent)
            
            return QPixmap.fromImage(result)
        except:
            return None
        
    def init_ui(self):
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.title_bar)  # 添加自定义标题栏
        
        # 内容容器 - 设置为透明背景
        content_widget = QWidget()
        content_widget.setStyleSheet("background: transparent;")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(30, 30, 30, 30)
        
        # 添加提取的黑色图片
        try:
            # 提取黑色部分
            black_pixmap = self.extract_black_parts("./9.10 1523/assets/Taekwondo.png")
            if black_pixmap and not black_pixmap.isNull():
                # 创建图片标签并居中
                image_label = QLabel()
                # 按比例缩放图片，保持宽高比
                scaled_pixmap = black_pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                image_label.setPixmap(scaled_pixmap)
                image_label.setAlignment(Qt.AlignCenter)
                
                # 添加图片到布局
                content_layout.addWidget(image_label, 0, Qt.AlignCenter)
                content_layout.addSpacing(-60)
        except Exception as e:
            print(f"图片处理错误: {e}")
        
        # 标题
        title_label = QLabel("跆拳道品势训练")
        title_label.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        title_label.setStyleSheet("""
            QLabel {
                color: rgb(23, 20, 53);
                background-color: transparent;
                padding: 10px 20px;
                border-radius: 8px;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(title_label)
        
        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("border: 1px solid rgba(255, 255, 255, 0.2); margin: 0px 50px;")
        separator.setFixedHeight(1)
        content_layout.addWidget(separator)
        
        # 添加一个负间距使按钮上移
        content_layout.addSpacing(-60)  # 负值会使后续内容上移

        # 按钮区域
        button_container = QWidget()
        button_container.setFixedWidth(int(self.width() * 0.35))
        button_layout = QVBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(20)
        
        # 开始按钮
        self.btn_start = QPushButton("开始训练")
        self.btn_start.setFixedHeight(60)
        self.btn_start.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: rgb(160, 199, 255);
                color: white;
                border-radius: 10px;
                padding: 12px;
                border: 2px solid rgb(160, 199, 255);
            }
            QPushButton:hover {
                background-color: rgb(140, 179, 235);
            }
            QPushButton:pressed {
                background-color: rgb(120, 159, 215);
            }
        """)
        
        # 历史记录按钮
        self.btn_history = QPushButton("历史记录")
        self.btn_history.setFixedHeight(60)
        self.btn_history.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.btn_history.setStyleSheet("""
            QPushButton {
               background-color: rgba(255, 255, 255, 0.9);
                color: rgb(160, 199, 255);
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
        
        button_layout.addWidget(self.btn_start)
        button_layout.addWidget(self.btn_history)
        content_layout.addWidget(button_container, 0, Qt.AlignCenter)
        
        main_layout.addWidget(content_widget)
        
        # 连接信号
        self.btn_start.clicked.connect(self.on_login_in)
        self.btn_history.clicked.connect(self.on_login_out)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.y() < self.title_bar.height():
            self.drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and hasattr(self, 'drag_pos'):
            self.move(event.globalPos() - self.drag_pos)
            event.accept()

    def on_login_in(self):
        self.main_window = PoseEstimationApp()
        self.main_window.show()
        self.hide()
        self.main_window.return_signal.connect(self.show)
    
    def on_login_out(self):
        self.history_window = HistoryWindow()
        self.history_window.show()
        self.hide()
        self.history_window.return_signal.connect(self.show)