from PyQt5.QtWidgets import QApplication
from login_window import LoginWindow

if __name__ == "__main__":
    app = QApplication([])
    
    # 创建并显示登录窗口
    login_window = LoginWindow()
    login_window.show()
    
    app.exec_()