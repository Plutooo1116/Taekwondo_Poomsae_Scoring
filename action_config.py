# 动作评分参数配置文件
# 可以根据实际测试结果调整各个阶段的参数

# ST-GCN动作类别到阶段名称的映射
ACTION_CLASS_TO_STAGE = {
    "preparation": "品势准备",
    "forward_stance_low_block": ["左转下格挡", "右转下格挡"],  # 可能对应多个阶段
    "forward_stance_punch": "前行步冲拳",
    "front_punch": "向前冲拳", 
    "low_block": "弓步下格挡",
    "middle_block": ["右转前行步中格挡", "左转前行步中格挡"],
    "rising_block": ["左转前行步上格挡", "转身前行步上格挡"],
    "front_kick_punch": "前踢前行步冲拳",
    "turn_forward_stance_low_block": "向后转弓步下格挡",
    "turn_forward_stance_punch": "弓步冲拳"
}

# 各阶段的评分参数配置
STAGE_EVALUATION_PARAMS = {
    "品势准备": {
        "threshold": 3.5,           # DTW距离阈值，越小越严格
        "oks_sigma": 0.25,          # OKS敏感度，越小越敏感
        "min_confidence": 0.15,     # 最小置信度要求
        "reference_csv": "./reference_data/action_0.csv",
        "weight": 1.0               # 该阶段在总分中的权重
    },
    
    "左转下格挡": {
        "threshold": 3.0,
        "oks_sigma": 0.2, 
        "min_confidence": 0.2,
        "reference_csv": "./reference_data/action_1.csv",
        "weight": 1.0
    },
    
    "前行步冲拳": {
        "threshold": 2.8,
        "oks_sigma": 0.18,
        "min_confidence": 0.2,
        "reference_csv": "./reference_data/action_2.csv", 
        "weight": 1.0
    },
    
    "右转下格挡": {
        "threshold": 2.8,
        "oks_sigma": 0.18,
        "min_confidence": 0.2,
        "reference_csv": "./reference_data/action_1.csv",
        "weight": 1.0
    },
    
    "转向前弓步下格挡": {
        "threshold": 2.8,
        "oks_sigma": 0.18,
        "min_confidence": 0.2,
        "reference_csv": "./reference_data/action_3.csv",
        "weight": 1.0
    },
    
    "向前冲拳": {
        "threshold": 2.8,
        "oks_sigma": 0.18,
        "min_confidence": 0.2,
        "reference_csv": "./reference_data/action_4.csv",
        "weight": 1.0
    },
    
    "右转前行步中格挡": {
        "threshold": 2.8,
        "oks_sigma": 0.18,
        "min_confidence": 0.2,
        "reference_csv": "./reference_data/action_5.csv",
        "weight": 1.0
    },
    
    "左转前行步中格挡": {
        "threshold": 2.8,
        "oks_sigma": 0.18,
        "min_confidence": 0.2,
        "reference_csv": "./reference_data/action_5.csv",
        "weight": 1.0
    },
    
    "弓步下格挡": {
        "threshold": 2.8,
        "oks_sigma": 0.18,
        "min_confidence": 0.2,
        "reference_csv": "./reference_data/action_3.csv",
        "weight": 1.0
    },
    
    "左转前行步上格挡": {
        "threshold": 2.8,
        "oks_sigma": 0.18,
        "min_confidence": 0.2,
        "reference_csv": "./reference_data/action_6.csv",
        "weight": 1.0
    },
    
    "前踢前行步冲拳": {
        "threshold": 2.8,
        "oks_sigma": 0.18,
        "min_confidence": 0.2,
        "reference_csv": "./reference_data/action_7.csv",
        "weight": 1.0
    },
    
    "转身前行步上格挡": {
        "threshold": 2.8,
        "oks_sigma": 0.18,
        "min_confidence": 0.2,
        "reference_csv": "./reference_data/action_6.csv",
        "weight": 1.0
    },
    
    "向后转弓步下格挡": {
        "threshold": 2.8,
        "oks_sigma": 0.18,
        "min_confidence": 0.2,
        "reference_csv": "./reference_data/action_8.csv",
        "weight": 1.0
    },
    
    "弓步冲拳": {
        "threshold": 2.8,
        "oks_sigma": 0.18,
        "min_confidence": 0.2,
        "reference_csv": "./reference_data/action_9.csv",
        "weight": 1.0
    },
    
    "结束动作": {
        "threshold": 2.8,
        "oks_sigma": 0.18,
        "min_confidence": 0.2,
        "reference_csv": "./reference_data/action_0.csv",
        "weight": 1.0
    }
}

# ST-GCN类别ID到动作名称的映射
STGCN_CLASS_NAMES = {
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

def get_stage_params(stage_name):
    """获取指定阶段的参数"""
    return STAGE_EVALUATION_PARAMS.get(stage_name, {
        "threshold": 3.0,
        "oks_sigma": 0.2,
        "min_confidence": 0.2,
        "reference_csv": None,
        "weight": 1.0
    })

def get_matching_stages(action_class_name):
    """根据ST-GCN分类结果获取对应的阶段列表"""
    mapping = ACTION_CLASS_TO_STAGE.get(action_class_name, [])
    if isinstance(mapping, str):
        return [mapping]
    elif isinstance(mapping, list):
        return mapping
    else:
        return []

def update_stage_params(stage_name, **kwargs):
    """更新指定阶段的参数"""
    if stage_name in STAGE_EVALUATION_PARAMS:
        STAGE_EVALUATION_PARAMS[stage_name].update(kwargs)
        print(f"已更新阶段 '{stage_name}' 的参数: {kwargs}")
    else:
        print(f"未找到阶段 '{stage_name}'")

def list_all_stages():
    """列出所有配置的阶段"""
    return list(STAGE_EVALUATION_PARAMS.keys())

def validate_reference_files():
    """验证所有参考文件是否存在"""
    import os
    missing_files = []
    
    for stage_name, params in STAGE_EVALUATION_PARAMS.items():
        ref_csv = params.get("reference_csv")
        if ref_csv and not os.path.exists(ref_csv):
            missing_files.append((stage_name, ref_csv))
    
    if missing_files:
        print("以下参考文件不存在:")
        for stage, file_path in missing_files:
            print(f"  阶段 '{stage}': {file_path}")
        return False
    else:
        print("所有参考文件验证通过")
        return True

# 使用示例
if __name__ == "__main__":
    print("动作评分参数配置")
    print("=" * 50)
    
    # 列出所有阶段
    print("配置的阶段:")
    for i, stage in enumerate(list_all_stages(), 1):
        params = get_stage_params(stage)
        print(f"{i:2d}. {stage}")
        print(f"    threshold={params['threshold']}, oks_sigma={params['oks_sigma']}")
        print(f"    min_confidence={params['min_confidence']}")
        print(f"    reference_csv={params['reference_csv']}")
    
    # 验证参考文件
    print("\n验证参考文件...")
    validate_reference_files()
    
    # 测试动作映射
    print("\nST-GCN动作映射测试:")
    for class_id, action_name in STGCN_CLASS_NAMES.items():
        stages = get_matching_stages(action_name)
        print(f"类别{class_id} ({action_name}) -> {stages}")
