import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局配置 =====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# AE 核心参数
TARGET_BRIGHTNESS = 225  # 10bit Gamma 空间 18% 灰目标值
ISO_RANGE = (100, 3200)  # ISO 范围
SHUTTER_RANGE = (1/60, 1/10000)  # 快门时间范围（s），1/60s~1/10000s
PID_PARAMS = {"Kp": 0.8, "Ki": 0.1, "Kd": 0.05}  # PID 系数
ROI_WEIGHT = {"center": 0.8, "edge": 0.2}  # 中心/边缘加权

# ===================== 1. 生成模拟场景图像（含不同亮度） =====================
def generate_scene_image(scene_type="normal", h=480, w=640):
    """
    生成不同场景的模拟图像（10bit Gamma 空间）
    :param scene_type: normal/backlight/lowlight/night
    :return: 10bit 图像（0~1023）、真实曝光参数（shutter, iso）
    """
    # 基础图像（随机噪声模拟纹理）
    #img = np.random.randint(0, 1023, (h, w), dtype=np.float32)
    img = np.random.randint(0, 1023, (h, w), dtype=np.uint16).astype(np.float32)
    # 不同场景的初始曝光
    if scene_type == "normal":
        shutter = 1/100  # 10ms
        iso = 100
        img = img * 0.4  # 初始亮度接近目标
    elif scene_type == "backlight":
        shutter = 1/200  # 5ms
        iso = 100
        img[:h//2, :] = img[:h//2, :] * 1.2  # 上半部分（天空）过曝
        img[h//2:, :] = img[h//2:, :] * 0.2  # 下半部分（主体）欠曝
    elif scene_type == "lowlight":
        shutter = 1/200  # 5ms
        iso = 100
        img = img * 0.1  # 整体欠曝
    elif scene_type == "night":
        shutter = 1/100  # 10ms
        iso = 200
        img = img * 0.15 + np.random.normal(0, 10, (h, w))  # 低光+噪点
    
    # 模拟 Sensor 曝光：亮度 = 基础亮度 × ISO × 快门时间
    img = img * iso * shutter * 1000  # 缩放系数适配范围
    img = np.clip(img, 0, 1023)
    return img, shutter, iso

# ===================== 2. AE 核心函数 =====================
def brightness_statistics(img, roi_mode="center_weight"):
    """
    亮度统计（支持全局/中心加权/多区域测光）
    :param img: 10bit 图像（Gamma 空间）
    :param roi_mode: global/center_weight/matrix
    :return: 加权亮度均值
    """
    h, w = img.shape
    
    if roi_mode == "global":
        return np.mean(img)
    
    elif roi_mode == "center_weight":
        # 中心区域：40% 画面（20%~80% 坐标）
        center_h_start, center_h_end = int(h*0.2), int(h*0.8)
        center_w_start, center_w_end = int(w*0.2), int(w*0.8)
        center_region = img[center_h_start:center_h_end, center_w_start:center_w_end]

        edge_parts = [
        img[:center_h_start, :].flatten(),  # 上边缘
        img[center_h_end:, :].flatten(),    # 下边缘
        img[:, :center_w_start].flatten(),  # 左边缘
        img[:, center_w_end:].flatten()     # 右边缘
        ]
        edge_region = np.concatenate(edge_parts)  # 一维拼接，无维度冲突
        
        # 加权计算
        center_mean = np.mean(center_region)
        edge_mean = np.mean(edge_region) if len(edge_region) > 0 else 0
        return center_mean * ROI_WEIGHT["center"] + edge_mean * ROI_WEIGHT["edge"]
    
    elif roi_mode == "matrix":
        # 16×16 矩阵测光（简化版）
        grid_h, grid_w = 16, 16
        grid_h_step = h // grid_h
        grid_w_step = w // grid_w
        grid_means = []
        for i in range(grid_h):
            for j in range(grid_w):
                grid = img[i*grid_h_step:(i+1)*grid_h_step, j*grid_w_step:(j+1)*grid_w_step]
                grid_means.append(np.mean(grid))
        # 中心 4×4 网格加权 0.7，其余 0.3
        center_grid_idx = [i*grid_w + j for i in range(6,10) for j in range(6,10)]
        center_mean = np.mean([grid_means[idx] for idx in center_grid_idx])
        edge_mean = np.mean([grid_means[idx] for idx in range(len(grid_means)) if idx not in center_grid_idx])
        return center_mean * 0.7 + edge_mean * 0.3

def pid_controller(current_brightness, target_brightness, pid_params, error_history):
    """
    PID 闭环控制计算曝光补偿量
    :param current_brightness: 当前亮度
    :param target_brightness: 目标亮度
    :param pid_params: Kp/Ki/Kd
    :param error_history: 历史误差列表
    :return: 曝光补偿量（>0 需增加曝光，<0 需减少）
    """
    # 计算当前误差
    error = target_brightness - current_brightness
    error_history.append(error)
    
    # 限制历史误差长度（避免积分饱和）
    if len(error_history) > 100:
        error_history = error_history[-100:]
    
    # P 项
    p_term = pid_params["Kp"] * error
    
    # I 项（积分）
    i_term = pid_params["Ki"] * np.sum(error_history)
    
    # D 项（微分）
    d_term = 0
    if len(error_history) >= 2:
        d_term = pid_params["Kd"] * (error_history[-1] - error_history[-2])
    
    # 总补偿量
    compensation = p_term + i_term + d_term
    return compensation, error_history

def adjust_exposure(compensation, current_shutter, current_iso):
    """
    根据补偿量调节快门/ISO（工程策略）
    :param compensation: 曝光补偿量（>0 加曝光）
    :param current_shutter: 当前快门时间（s）
    :param current_iso: 当前 ISO
    :return: 新快门、新 ISO
    """
    # 计算需要的曝光增益（补偿量 / 目标亮度 = 增益倍数）
    gain = 1 + (compensation / TARGET_BRIGHTNESS)
    gain = np.clip(gain, 0.1, 10)  # 限制增益范围
    
    # 调节策略：优先调快门（无噪点），快门到极限后调 ISO
    new_shutter = current_shutter * gain
    new_iso = current_iso
    
    # 快门上限/下限检查
    if new_shutter > SHUTTER_RANGE[0]:  # 快门最长 1/60s
        excess_gain = new_shutter / SHUTTER_RANGE[0]
        new_shutter = SHUTTER_RANGE[0]
        new_iso = current_iso * excess_gain
    elif new_shutter < SHUTTER_RANGE[1]:  # 快门最短 1/10000s
        deficit_gain = new_shutter / SHUTTER_RANGE[1]
        new_shutter = SHUTTER_RANGE[1]
        new_iso = current_iso * deficit_gain
    
    # ISO 范围限制
    new_iso = np.clip(new_iso, ISO_RANGE[0], ISO_RANGE[1])
    
    return new_shutter, new_iso

def apply_exposure(img, current_shutter, current_iso, new_shutter, new_iso):
    """
    模拟 Sensor 应用新曝光参数后的图像亮度变化
    :param img: 原始图像
    :param current_shutter/current_iso: 旧参数
    :param new_shutter/new_iso: 新参数
    :return: 新亮度图像
    """
    # 曝光增益 = (新快门/旧快门) × (新ISO/旧ISO)
    exposure_gain = (new_shutter / current_shutter) * (new_iso / current_iso)
    new_img = img * exposure_gain
    # 加入高 ISO 噪点
    iso_noise = np.random.normal(0, (new_iso - 100)/10, img.shape) if new_iso > 100 else 0
    new_img = new_img + iso_noise
    return np.clip(new_img, 0, 1023)

# ===================== 3. 验证函数 =====================
def verify_ae_effect(brightness_history, shutter_history, iso_history):
    """
    验证 AE 效果：亮度收敛性、参数稳定性
    """
    plt.figure(figsize=(15, 5))
    
    # 1. 亮度收敛曲线
    plt.subplot(1, 3, 1)
    plt.plot(brightness_history, label="当前亮度")
    plt.axhline(y=TARGET_BRIGHTNESS, color='r', linestyle='--', label="目标亮度（18%灰）")
    plt.xlabel("迭代次数")
    plt.ylabel("亮度值（10bit Gamma）")
    plt.title("AE 亮度收敛曲线")
    plt.legend()
    plt.grid(True)
    
    # 2. 快门时间变化
    plt.subplot(1, 3, 2)
    plt.plot(shutter_history, label="快门时间（s）", color='g')
    plt.xlabel("迭代次数")
    plt.ylabel("快门时间（s）")
    plt.title("快门时间调节曲线")
    plt.grid(True)
    
    # 3. ISO 变化
    plt.subplot(1, 3, 3)
    plt.plot(iso_history, label="ISO", color='orange')
    plt.xlabel("迭代次数")
    plt.ylabel("ISO")
    plt.title("ISO 调节曲线")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 量化指标
    final_brightness = brightness_history[-1]
    brightness_error = abs(final_brightness - TARGET_BRIGHTNESS) / TARGET_BRIGHTNESS * 100
    print(f"\n=== AE 效果验证 ===")
    print(f"最终亮度：{final_brightness:.2f}（目标：{TARGET_BRIGHTNESS}）")
    print(f"亮度误差率：{brightness_error:.2f}%")
    print(f"最终快门：{shutter_history[-1]:.6f}s，最终 ISO：{iso_history[-1]:.0f}")
    return brightness_error

# ===================== 4. 场景适配策略（逆光/低光/夜景） =====================
def scene_adaptive_ae(img, brightness, shutter, iso, scene_type):
    """
    场景自适应 AE 策略
    """
    h, w = img.shape
    new_shutter, new_iso = shutter, iso
    
    if scene_type == "backlight":
        # 逆光：提升曝光补偿，压低高光
        highlight_mask = img > 800  # 高光区域（10bit）
        if np.sum(highlight_mask) > 0.1 * h * w:  # 高光占比>10%
            # 提升快门（增加曝光），但限制高光过曝
            new_shutter = min(shutter * 1.2, SHUTTER_RANGE[0])
            new_iso = iso  # 不提升 ISO 避免噪点
    elif scene_type == "lowlight":
        # 低光：优先提 ISO，再延长快门
        if brightness < TARGET_BRIGHTNESS * 0.5:
            new_iso = min(iso * 2, ISO_RANGE[1])
            if new_iso >= ISO_RANGE[1]:
                new_shutter = min(shutter * 1.5, SHUTTER_RANGE[0])
    elif scene_type == "night":
        # 夜景：长快门+中高 ISO，加入防频闪
        new_shutter = SHUTTER_RANGE[0]  # 50Hz 防频闪（1/60s，不超出范围）
        new_iso = min(iso * 3, ISO_RANGE[1])
    
    return new_shutter, new_iso

# ===================== 5. 主流程 =====================
if __name__ == "__main__":
    # 选择测试场景：normal/backlight/lowlight/night
    SCENE_TYPE = "lowlight"
    
    # 初始化
    img, shutter, iso = generate_scene_image(SCENE_TYPE)
    brightness_history = []
    shutter_history = [shutter]
    iso_history = [iso]
    error_history = []
    iter_num = 10  # AE 迭代次数（实际项目中是实时闭环）
    
    # AE 闭环迭代
    for i in range(iter_num):
        # 1. 亮度统计
        current_brightness = brightness_statistics(img, roi_mode="center_weight")
        brightness_history.append(current_brightness)
        
        # 2. PID 计算补偿量
        compensation, error_history = pid_controller(
            current_brightness, TARGET_BRIGHTNESS, PID_PARAMS, error_history
        )
        
        # 3. 调节曝光参数
        new_shutter, new_iso = adjust_exposure(compensation, shutter, iso)
        
        # 4. 场景自适应优化
        new_shutter, new_iso = scene_adaptive_ae(img, current_brightness, new_shutter, new_iso, SCENE_TYPE)
        
        # 5. 应用新曝光参数
        img = apply_exposure(img, shutter, iso, new_shutter, new_iso)
        
        # 6. 更新参数
        shutter, iso = new_shutter, new_iso
        shutter_history.append(shutter)
        iso_history.append(iso)
        
        print(f"迭代 {i+1}：亮度={current_brightness:.2f}，补偿量={compensation:.2f}，快门={shutter:.6f}s，ISO={iso:.0f}")
    
    # 验证 AE 效果
    verify_ae_effect(brightness_history, shutter_history, iso_history)
    
    # 显示最终图像
    plt.figure(figsize=(8, 6))
    plt.imshow((img / 1023 * 255).astype(np.uint8), cmap='gray')
    plt.title(f"AE 调节后图像（场景：{SCENE_TYPE}）")
    plt.axis('off')
    plt.show()