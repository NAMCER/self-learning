import cv2
import numpy as np
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 模拟生成带故障的 RAW 图（模拟 Sensor 输出） =====================
def generate_faulty_raw():
    """
    生成 512x512 的 16bit RAW 图（模拟 RGGB Bayer 格式）
    加入：
    - 坏点（死点+热点）
    - 黑电平偏移（所有像素+32）
    - 镜头暗角（中心亮、边缘暗的渐变）
    """
    # 基础 RAW 图（模拟正常感光，值范围 0~1023，10bit 数据）
    raw = np.random.randint(0, 1024, (512, 512), dtype=np.uint16)
    
    # 1. 加入坏点（实际项目中坏点坐标由 Sensor 厂提供坏点表）
    # 死点（值=0）：随机选10个位置
    dead_pixels = np.random.choice(512*512, 10, replace=False)
    dead_y = dead_pixels // 512
    dead_x = dead_pixels % 512
    raw[dead_y, dead_x] = 0
    
    # 热点（值=1023）：随机选10个位置
    hot_pixels = np.random.choice(512*512, 10, replace=False)
    hot_y = hot_pixels // 512
    hot_x = hot_pixels % 512
    raw[hot_y, hot_x] = 1023
    
    # 2. 加入黑电平偏移（实际项目中黑电平值由 Sensor 规格书定，通常 16/32/64）
    black_level = 32
    raw = raw + black_level  # 所有像素抬高 32，暗部不再是 0
    
    # 3. 加入镜头暗角（LSC 故障）：生成中心亮、边缘暗的衰减矩阵
    # 计算每个像素到中心的距离
    y, x = np.meshgrid(np.arange(512), np.arange(512), indexing='ij')
    center_y, center_x = 256, 256
    distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    max_distance = np.sqrt(256**2 + 256**2)
    # 衰减系数：中心=1（无衰减），边缘=0.5（衰减50%）
    lsc_attn = 1 - 0.5 * (distance / max_distance)
    # 应用暗角（模拟镜头进光衰减）
    raw = (raw * lsc_attn).astype(np.uint16)
    
    return raw, black_level

# 生成故障 RAW 图
raw_faulty, black_level = generate_faulty_raw()
# 归一化到 0~255 用于显示（实际处理不做归一化，保持 16bit）
raw_show = (raw_faulty / np.max(raw_faulty) * 255).astype(np.uint8)

# 显示原始故障图
plt.figure(figsize=(8, 6))
plt.imshow(raw_show, cmap='gray')
plt.title('原始故障 RAW 图（含坏点+黑电平+镜头暗角）')
plt.axis('off')
plt.show()