import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import filedialog
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort  # 导入 ONNX Runtime
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 关键点连接关系
CONNECTIONS = [
    (0, 14),   # 鼻子-脖子
    (1, 14),   # 左肩-脖子
    (2, 14),   # 右肩-脖子
    (1, 3),   # 左肩-左肘
    (2, 4),   # 右肩-右肘
    (3, 5),   # 左肘-左手腕
    (4, 6),   # 右肘-右手腕
    (1, 7),   # 左肩-左髋
    (2, 8),   # 右肩-右髋
    (7, 9),   # 左髋-左膝
    (8, 10),  # 右髋-右膝
    (9, 11),  # 左膝-左脚踝
    (10, 12), # 右膝-右脚踝
    (7, 13),  # 左髋-骨盆
    (8, 13),  # 右髋-骨盆
    (7, 8)    # 左髋-右髋
]

# 右侧关节索引 (需要镜像恢复的关节)
RIGHT_JOINT_INDICES = [2, 4, 6, 8, 10, 12]  # 右肩、右肘、右手腕、右髋、右膝、右脚踝

# 动作类别映射
ACTION_CLASSES = {
    0: "forward_stance_low_block",
    1: "forward_stance_punch",
    2: "front_kick_punch",
    3: "front_punch",
    4: "low_block",
    5: "middle_block",
    6: "preparation",
    7: "rising_block",
    8: "turn_forward_stance_low_block",
    9: "turn_forward_stance_punch"
}

# 初始化ONNX模型
def init_onnx_model(model_path='ST_GCN_NEW.onnx'):
    """初始化ONNX模型会话"""
    # 创建ONNX Runtime会话
    session = ort.InferenceSession(model_path)
    
    # 打印输入输出信息用于调试
    print("ONNX模型输入信息:")
    for i, input_info in enumerate(session.get_inputs()):
        print(f"  输入 {i}: 名称={input_info.name}, 形状={input_info.shape}, 类型={input_info.type}")
    
    print("ONNX模型输出信息:")
    for i, output_info in enumerate(session.get_outputs()):
        print(f"  输出 {i}: 名称={output_info.name}, 形状={output_info.shape}, 类型={output_info.type}")
    
    return session

# 处理窗口数据 - ONNX版本
def process_window_onnx(coord_window, motion_window, session):
    """处理窗口数据，匹配ONNX模型的输入格式"""
    # 准备ONNX模型的输入
    # coord_window 形状: (15, 3, T) -> 需要转换为 (1, 3, T, 15)
    coord_input = coord_window.transpose(1, 2, 0)[np.newaxis, ...].astype(np.float32)
    
    # motion_window 形状: (15, 2, T) -> 需要转换为 (1, 2, T, 15)
    motion_input = motion_window.transpose(1, 2, 0)[np.newaxis, ...].astype(np.float32)
    
    # 获取ONNX模型的输入名称
    input_names = [input.name for input in session.get_inputs()]
    
    # 运行ONNX模型推理
    outputs = session.run(
        None, 
        {
            input_names[0]: coord_input,
            input_names[1]: motion_input
        }
    )
    
    # 输出通常是logits，需要应用softmax
    logits = outputs[0]
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    predicted_class = np.argmax(probabilities, axis=1)[0]
    
    return predicted_class, probabilities[0]

def load_csv_data(csv_path):
    """加载CSV骨架数据并恢复镜像"""
    df = pd.read_csv(csv_path)
    
    # 提取关键点数据 (frame_num + 15 * 3 keypoints)
    frames = []
    for _, row in df.iterrows():
        frame_num = int(row['frame_num'])
        kps = row[1:].values.reshape(15, 3)  # 转换为(15,3)数组: x,y,conf
        
        # 恢复镜像：右侧关节x坐标取反
        kps[RIGHT_JOINT_INDICES, 0] *= -1
        
        frames.append({
            'frame_num': frame_num,
            'keypoints': kps
        })
    
    print(f"已加载 {len(frames)} 帧骨架数据，并恢复镜像")
    return frames

def create_skeleton_animation(frames, session, output_gif=None):
    """
    创建带动作分类的2D骨架动画 - ONNX版本
    参数:
        frames: 骨骼帧数据列表
        session: ONNX Runtime会话
        output_gif: 如需保存GIF则传入路径
    返回:
        matplotlib动画对象
    """
    # 初始化图形界面（优化后的布局）
    fig = plt.figure(figsize=(16, 8), facecolor='white')
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[4, 1])
    
    # 主骨架图
    ax_skeleton = fig.add_subplot(gs[0, 0])
    ax_skeleton.set_title('太极一章骨骼动画', pad=20)
    ax_skeleton.set_xlabel('X轴 (骨盆中心为原点)')
    ax_skeleton.set_ylabel('Y轴')
    ax_skeleton.invert_yaxis()
    
    # 信息显示区域
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis('off')
    
    # 控制面板区域
    ax_control = fig.add_subplot(gs[1, :])
    ax_control.axis('off')
    
    # 初始化绘图元素
    points = ax_skeleton.scatter([], [], c='blue', s=50, alpha=0.8)
    lines = [ax_skeleton.plot([], [], 'r-', linewidth=1.5)[0] for _ in CONNECTIONS]
    
    # 文本信息
    frame_text = ax_info.text(0.1, 0.9, '准备就绪...', 
                            fontsize=10, transform=ax_info.transAxes)
    action_text = ax_info.text(0.1, 0.8, '动作类别: ', 
                             fontsize=12, weight='bold', transform=ax_info.transAxes)
    
    # 置信度柱状图
    ax_probs = ax_info.inset_axes([0.1, 0.1, 0.8, 0.6])
    ax_probs.set_title('动作置信度', pad=10)
    ax_probs.set_ylim(0, 1)
    ax_probs.set_xticks(range(len(ACTION_CLASSES)))
    ax_probs.set_xticklabels([str(i) for i in range(len(ACTION_CLASSES))], rotation=45)
    bars = ax_probs.bar(range(len(ACTION_CLASSES)), [0]*len(ACTION_CLASSES), 
                       color='lightblue', edgecolor='black')
    
    # 控制按钮
    play_ax = ax_control.inset_axes([0.4, 0.6, 0.2, 0.3])
    play_button = Button(play_ax, '播放/暂停', color='lightgreen')
    
    # 进度条
    slider_ax = ax_control.inset_axes([0.1, 0.2, 0.8, 0.3])
    frame_slider = Slider(slider_ax, '帧进度', 0, len(frames)-1, valinit=0, valstep=1)

    # 预计算分类结果
    window_size = 20
    window_stride = 10
    window_predictions = []
    
    # 准备数据（确保3通道）
    coords = np.array([f['keypoints'][:, :2] for f in frames])  # (T, 15, 2)
    confs = np.array([f['keypoints'][:, 2] for f in frames])    # (T, 15)
    motions = np.zeros_like(coords)
    motions[1:] = coords[1:] - coords[:-1]
    
    # 合并为3通道 (x, y, conf)
    coords_3ch = np.stack([coords[:, :, 0], coords[:, :, 1], confs], axis=2)  # (T, 15, 3)

    # 处理每个窗口
    print("正在计算动作分类...")
    for start_idx in range(0, len(frames) - window_size + 1, window_stride):
        end_idx = start_idx + window_size
        
        coord_window = coords_3ch[start_idx:end_idx].transpose(1, 2, 0)  # (15, 3, T)
        motion_window = motions[start_idx:end_idx].transpose(1, 2, 0)    # (15, 2, T)
        
        # 使用ONNX模型进行推理
        pred_class, probs = process_window_onnx(coord_window, motion_window, session)
        
        window_predictions.append({
            'start_frame': start_idx,
            'end_frame': end_idx - 1,
            'predicted_class': pred_class,
            'probabilities': probs
        })
    
    print(f"已完成 {len(window_predictions)} 个窗口的分类计算")

    # 动画更新函数
    def update(frame_idx):
        frame = frames[frame_idx]
        keypoints = frame['keypoints']
        
        # 获取当前帧所有可见的关键点坐标
        visible_keypoints = keypoints[keypoints[:, 2] > 0.1, :2]
        
        # 如果有可见的关键点，计算数据的范围并设置坐标轴
        if len(visible_keypoints) > 0:
            x_min, y_min = np.min(visible_keypoints, axis=0)
            x_max, y_max = np.max(visible_keypoints, axis=0)
            
            # 添加一些边距，使骨架不会紧贴坐标轴边缘
            margin = 0.1 * max(x_max - x_min, y_max - y_min)
            x_min -= margin
            x_max += margin
            y_min -= margin
            y_max += margin
            
            # 设置坐标轴范围
            ax_skeleton.set_xlim(x_min, x_max)
            ax_skeleton.set_ylim(y_max, y_min)
        
        # 更新骨架
        visible = keypoints[:, 2] > 0.1
        points.set_offsets(keypoints[visible, :2])
        
        # 更新骨骼连接线
        for i, (start, end) in enumerate(CONNECTIONS):
            if keypoints[start, 2] > 0.1 and keypoints[end, 2] > 0.1:
                lines[i].set_data([keypoints[start, 0], keypoints[end, 0]],
                                [keypoints[start, 1], keypoints[end, 1]])
                lines[i].set_visible(True)
            else:
                lines[i].set_visible(False)
        
        # 更新分类信息
        current_action = "未知"
        current_probs = np.zeros(len(ACTION_CLASSES))
        
        for win in window_predictions:
            if win['start_frame'] <= frame_idx <= win['end_frame']:
                current_action = ACTION_CLASSES[win['predicted_class']]
                current_probs = win['probabilities']
                
                # 高亮当前预测动作
                for i, bar in enumerate(bars):
                    bar.set_height(current_probs[i])
                    bar.set_color('red' if i == win['predicted_class'] else 'lightblue')
                break
        
        action_text.set_text(f"动作类别: {current_action}")
        frame_text.set_text(
            f"帧: {frame_idx+1}/{len(frames)}\n"
            f"骨盆坐标: ({keypoints[13,0]:.2f}, {keypoints[13,1]:.2f})"
        )
        
        # 返回所有需要更新的绘图元素（不包括bars，因为对它们的修改已经自动生效）
        return [points] + lines + [frame_text, action_text]

    # 事件处理函数
    def on_slider_change(val):
        if ani.event_source:
            ani.event_source.stop()
        update(int(val))
        fig.canvas.draw_idle()
    
    def on_button_click(event):
        if ani.event_source and ani.event_source.is_alive():
            ani.event_source.stop()
        else:
            ani.event_source.start()
    
    def on_key_press(event):
        if event.key == ' ':
            on_button_click(None)
    
    # 绑定事件
    frame_slider.on_changed(on_slider_change)
    play_button.on_clicked(on_button_click)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # 创建动画
    ani = FuncAnimation(
        fig, update, frames=len(frames),
        interval=100, blit=False, repeat=True
    )
    
    plt.tight_layout()
    plt.show()
    return ani

def browse_file():
    """打开文件选择对话框"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(
        title="选择CSV文件",
        filetypes=[("CSV文件", "*.csv")]
    )
    return file_path

if __name__ == "__main__":
    # 初始化ONNX模型
    print("正在初始化ONNX模型...")
    onnx_session = init_onnx_model('ST_GCN_NEW.onnx')  # 替换为你的ONNX模型路径
    print("ONNX模型初始化完成")
    
    # 让用户选择文件
    csv_path = browse_file()
    if not csv_path:
        print("未选择文件，程序退出。")
        sys.exit()

    # 加载数据并创建动画
    skeleton_frames = load_csv_data(csv_path)
    
    # 创建动画
    animation = create_skeleton_animation(skeleton_frames, onnx_session)