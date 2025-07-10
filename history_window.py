## 历史记录——分数详情

import os
import json
import time
from datetime import datetime
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QTableWidget, QTableWidgetItem, QScrollArea,QMessageBox,QDesktopWidget)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QPainter
from simple_video_player import SimpleVideoPlayer

class HistoryWindow(QWidget):
    return_signal = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self._init_ui()  # 先初始化UI组件
        self._setup_data()  # 然后初始化数据
        self._connect_signals()  # 最后连接信号
        
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
        """只初始化UI元素，不涉及数据加载"""
        self.setWindowTitle("成长记录")
        # 获取屏幕尺寸并设置窗口大小
        screen = QDesktopWidget().screenGeometry()
        self.setMinimumSize(int(screen.width()*0.5), int(screen.height()*0.6))
        
        try:
            self.setWindowIcon(QIcon("./9.10 1523/assets/Taekwondo.png"))
        except:
            pass

        # 主布局
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.main_layout)    

        self.btn_return = QPushButton("返回")
        self.btn_return.clicked.connect(self.on_return_clicked)
        self.main_layout.addWidget(self.btn_return, alignment=Qt.AlignLeft | Qt.AlignTop)  # 靠左上对齐
        self.btn_return.setStyleSheet("""
            QPushButton {
                min-width: 120px;  /* 增加宽度 */
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

        # 标题
        title_label = QLabel("训练历史记录")
        title_label.setStyleSheet("""
            font-size: 24px; 
            font-weight: bold;
            color: black;
            border-radius: 10px;
            padding: 10px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(title_label)
        self.main_layout.addSpacing(20)
        
        # 表格 - 现在有6列
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["日期", "动作", "分数", "详情", "视频", "删除"])
        self.table.setRowCount(0)
        self.table.setStyleSheet("font-size: 18px;")
        self.table.horizontalHeader().setStyleSheet("font-size: 18px; font-weight: bold;")
        self.table.verticalHeader().setDefaultSectionSize(50)
        # 设置各列宽度
        self.table.setColumnWidth(0, int(screen.width()*0.1))  # 日期列宽一些
        self.table.setColumnWidth(1, int(screen.width()*0.1))   # 动作列
        self.table.setColumnWidth(2, int(screen.width()*0.05))  # 分数列
        self.table.setColumnWidth(3, int(screen.width()*0.1))   # 详情列
        self.table.setColumnWidth(4, int(screen.width()*0.08))   # 视频列
        self.table.setColumnWidth(5, int(screen.width()*0.08))  # 删除列
        
        # 设置表格内容居中
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)

        # 详情区域 - 放大字体
        self.detail_label = QLabel()
        self.detail_label.setStyleSheet("""
            font-size: 20px; 
            background-color: rgba(255, 255, 255, 0.5);
            padding: 12px;
            line-height: 1.5;
            border-radius: 5px;
        """)
        self.detail_label.setWordWrap(True)
        self.detail_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        scroll_area = QScrollArea()
        scroll_area.setStyleSheet("""
            background-color: transparent;
            border: none;
        """)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.detail_label)
        
        # 评分详情标题 - 也放大字体
        detail_title = QLabel("评分详情:")
        detail_title.setStyleSheet("""
            font-size: 20px; 
            font-weight: bold;
            color: black;
            border-radius: 5px;
            padding: 5px 10px;
        """)
        
        self.main_layout.addWidget(self.table)
        self.main_layout.addWidget(detail_title)
        self.main_layout.addWidget(scroll_area)
    
    def _setup_data(self):
        """初始化数据相关设置"""
        self.history_data = []
        self.max_records = 20 #最大保存记录数
        QTimer.singleShot(0, self._async_load_data)  # 异步加载数据
    
    def _connect_signals(self):
        """连接信号与槽"""
        self.btn_return.clicked.connect(self.on_return_clicked) 
    
    def _async_load_data(self):
        """异步加载历史数据"""
        try:
            if os.path.exists("history_data.json"):
                with open("history_data.json", "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                    # 按日期倒序排序
                    self.history_data = sorted(
                        loaded_data,
                        key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d %H:%M:%S"),
                        reverse=True
                    )
                    # 只保留最新的20条记录
                    if len(self.history_data) > self.max_records:
                        self.history_data = self.history_data[:self.max_records]
                    self._populate_table()
        except Exception as e:
            print(f"加载历史记录失败: {e}")
    
    def _populate_table(self):
        """填充表格数据"""
        self.table.setRowCount(0)  # 清空表格
        # 按日期倒序显示（最新的在最上面）
        for row, record in enumerate(self.history_data):
            self.table.insertRow(row)
            
            # 日期列
            date_item = QTableWidgetItem(record["date"])
            date_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 0, date_item)
            
            # 动作列的显示逻辑
            action_name = "品势一章"  # 默认值
            if "stage_scores" in record and len(record["stage_scores"]) > 0:
                # 从第一个阶段分数中提取动作名称
                first_score = record["stage_scores"][0]
                if ":" in first_score:
                    action_name = first_score.split(":")[0].strip()
                    # 如果是品势一章的完整记录
                    if len(record["stage_scores"]) >= 18 and "品势准备" in action_name:
                        action_name = "品势一章"
                else:
                    action_name = first_score
            
            action_item = QTableWidgetItem(action_name)
            action_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 1, action_item)
            
            score_item = QTableWidgetItem(f"{record['total_score']}")
            score_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 2, score_item)
            
            # 详情按钮
            btn_detail = QPushButton("查看详情")
            btn_detail.setStyleSheet("font-size: 18px;")
            btn_detail.clicked.connect(lambda _, r=row: self.show_details(r))
            self.table.setCellWidget(row, 3, btn_detail)

             # 视频按钮
            btn_video = QPushButton("播放")
            btn_video.setStyleSheet("font-size: 18px;")
            btn_video.clicked.connect(lambda _, r=row: self.view_video_by_record(r))
            self.table.setCellWidget(row, 4, btn_video)
            # 如果没有视频文件，禁用按钮
            if ("video_file" not in record or 
                not record["video_file"] or 
                record["video_file"] == "无视频文件" or 
                not os.path.exists(os.path.join("training_videos", record["video_file"]))):
                btn_video.setEnabled(False)
            
            # 删除按钮
            btn_delete = QPushButton("删除")
            btn_delete.setStyleSheet("font-size: 18px;")
            btn_delete.clicked.connect(lambda _, r=row: self._delete_record(r))
            self.table.setCellWidget(row, 5, btn_delete)

    def _delete_record(self, row):
        """删除指定行记录"""
        if 0 <= row < len(self.history_data):
            # 确认对话框
            reply = QMessageBox.question(
                self, 
                '确认删除',
                '确定要删除这条记录吗?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # 从数据中删除
                del self.history_data[row]
                # 刷新表格
                self._populate_table()
                # 保存到文件
                self.save_to_file()
    
    def add_record(self, date, total_score, stage_scores, video_file=None, csv_file=None):
        """添加新记录到历史数据并更新表格"""
        # 确保所有参数都有合理的默认值
        video_file = video_file if video_file is not None else "无视频文件"
        csv_file = csv_file if csv_file is not None else "无数据文件"
        
        # 确保stage_scores是列表且每个元素都是字符串
        if not isinstance(stage_scores, list):
            stage_scores = []
        stage_scores = [str(score) for score in stage_scores]
        
        record = {
            "date": str(date),
            "total_score": str(total_score),
            "stage_scores": stage_scores,
            "video_file": str(video_file),
            "csv_file": str(csv_file)
        }
        
        # 将新记录添加到列表开头
        self.history_data.insert(0, record)
        
        # 如果超过最大记录数，移除最旧的记录
        if len(self.history_data) > self.max_records:
            self.history_data = self.history_data[:self.max_records]
        
        # 刷新表格显示并保存
        self._populate_table()
        self.save_to_file()
    
    def show_details(self, row):
        """显示详细评分"""
        if 0 <= row < len(self.history_data):
            data = self.history_data[row]
            details = f"日期: {data['date']}\n"
            details += f"总分: {data['total_score']}\n\n"
            details += "各阶段评分:\n"
            
            # 检查是否是品势一章的完整记录
            is_full_poomsae = len(data["stage_scores"]) >= 2  # 品势一章有18个阶段
            
            if is_full_poomsae:
                # 品势一章的固定阶段顺序
                poomsae_stages = [
                    "品势准备",
                    "1. 左转下格挡",
                    "2. 前行步冲拳",
                    "3. 右转下格挡",
                    "4. 前行步冲拳",
                    "5. 转向前弓步下格挡冲拳",
                    "6. 右转前行步中格挡",
                    "7. 前行步冲拳",
                    "8. 转身前行步中格挡",
                    "9. 前行步冲拳",
                    "10. 弓步下格挡冲拳",
                    "11. 左转前行步上格挡",
                    "12. 前踢前行步冲拳",
                    "13. 转身前行步上格挡",
                    "14. 前踢前行步冲拳",
                    "15. 向后转弓步下格挡",
                    "16. 弓步冲拳（哈）",
                    "结束动作"
                ]
                
                # 确保阶段数量和顺序匹配
                for i, stage in enumerate(poomsae_stages):
                    if i < len(data["stage_scores"]):
                        # 分割阶段名称和分数
                        score = data["stage_scores"][i]
                        if ":" in score:
                            _, stage_score = score.split(":", 1)
                            details += f"  - {stage}: {stage_score.strip()}\n"
                        else:
                            details += f"  - {stage}: {score}\n"
            else:
                # 单个动作训练，直接显示所有分数
                for score in data["stage_scores"]:
                    if ":" in score:
                        stage_name, stage_score = score.split(":", 1)
                        details += f"  - {stage_name.strip()}: {stage_score.strip()}\n"
                    else:
                        details += f"  - {score}\n"
            
            self.detail_label.setText(details)
    
    def save_to_file(self):
        """将历史记录保存到文件"""
        try:
            with open("history_data.json", "w", encoding="utf-8") as f:
                json.dump(self.history_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"保存历史记录失败: {e}")

    def view_video_by_record(self, row):
        """查看指定记录的视频"""
        if 0 <= row < len(self.history_data):
            record = self.history_data[row]
            if "video_file" in record and record["video_file"] and record["video_file"] != "无视频文件":
                video_path = os.path.join("training_videos", record["video_file"])
                if os.path.exists(video_path):
                    self._play_video(video_path)
                else:
                    QMessageBox.warning(self, "提示", "视频文件不存在")
            else:
                QMessageBox.warning(self, "提示", "该记录没有关联的视频文件")

    def _play_video(self, video_path):
        """播放视频的通用方法"""
        if not os.path.exists(video_path):
            QMessageBox.warning(self, "提示", f"视频文件不存在: {video_path}")
            return
        
        # 创建视频播放窗口
        self.video_window = SimpleVideoPlayer(video_path)
        self.video_window.show()

    def on_return_clicked(self):
            self.return_signal.emit()
            self.close()