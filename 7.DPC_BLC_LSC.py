import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局配置 =====================
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 固定参数（模拟真实 Sensor 规格）
RAW_SIZE = (512, 512)  # RAW 图尺寸
BLACK_LEVEL = 32       # 黑电平值（Sensor 规格书提供）
RAW_MAX_VALUE = 1023   # 10bit RAW 最大值

# ===================== 1. 生成带故障的 RAW 图 =====================
def generate_faulty_raw():
    """
    生成模拟 RGGB Bayer 格式的 16bit RAW 图
    加入故障：坏点（死点+热点）、黑电平偏移、镜头暗角
    """
    # 生成基础随机 RAW 图（模拟 Sensor 原始感光数据）
    raw = np.random.randint(0, RAW_MAX_VALUE, RAW_SIZE, dtype=np.uint16)
    
    # 1. 加入坏点（实际项目中坏点坐标由 Sensor 厂提供坏点表）
    # 死点（值=0）：随机选10个位置
    dead_pixels = np.random.choice(RAW_SIZE[0]*RAW_SIZE[1], 10, replace=False)
    dead_y = dead_pixels // RAW_SIZE[1]
    dead_x = dead_pixels % RAW_SIZE[1]
    raw[dead_y, dead_x] = 0
    
    # 热点（值=RAW_MAX_VALUE）：随机选10个位置
    hot_pixels = np.random.choice(RAW_SIZE[0]*RAW_SIZE[1], 10, replace=False)
    hot_y = hot_pixels // RAW_SIZE[1]
    hot_x = hot_pixels % RAW_SIZE[1]
    raw[hot_y, hot_x] = RAW_MAX_VALUE
    
    # 2. 加入黑电平偏移（所有像素抬高 BLACK_LEVEL）
    raw = raw + BLACK_LEVEL
    
    # 3. 加入镜头暗角（生成中心亮、边缘暗的衰减矩阵）
    y, x = np.meshgrid(np.arange(RAW_SIZE[0]), np.arange(RAW_SIZE[1]), indexing='ij')
    center_y, center_x = RAW_SIZE[0]//2, RAW_SIZE[1]//2
    distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    max_distance = np.sqrt((RAW_SIZE[0]//2)**2 + (RAW_SIZE[1]//2)**2)
    lsc_attn = 1 - 0.5 * (distance / max_distance)  # 边缘衰减 50%
    raw = (raw * lsc_attn).astype(np.uint16)
    
    return raw

# ===================== 2. ISP 前处理核心函数 =====================
def blc_correction(raw, black_level):
    """
    黑电平校正（实际项目中第一步执行）
    :param raw: 输入 16bit RAW 图
    :param black_level: 黑电平值
    :return: 校正后 RAW 图
    """
    # 转 int32 防止减法溢出，截断负数为 0
    raw_corrected = raw.astype(np.int32) - black_level
    raw_corrected = np.maximum(raw_corrected, 0)
    return raw_corrected.astype(np.uint16)

def dpc_correction(raw):
    """
    坏点校正（实际项目中第二步执行）
    :param raw: 输入 16bit RAW 图（已做 BLC）
    :return: 校正后 RAW 图
    """
    raw_corrected = raw.copy()
    h, w = raw.shape
    
    # 1. 检测坏点：值=0（死点）或 值=RAW_MAX_VALUE（热点）
    bad_pixel_mask = (raw == 0) | (raw == RAW_MAX_VALUE)
    y_coords, x_coords = np.where(bad_pixel_mask)
    
    # 2. 遍历坏点，用 3x3 窗口有效像素均值替换
    for y, x in zip(y_coords, x_coords):
        # 确定 3x3 窗口范围（防止越界）
        y_start = max(0, y-1)
        y_end = min(h, y+2)
        x_start = max(0, x-1)
        x_end = min(w, x+2)
        
        # 提取窗口内像素，排除其他坏点
        window = raw[y_start:y_end, x_start:x_end]
        window_valid = window[(window != 0) & (window != RAW_MAX_VALUE)]
        
        # 用有效像素均值替换坏点
        if len(window_valid) > 0:
            raw_corrected[y, x] = np.mean(window_valid).astype(np.uint16)
    
    return raw_corrected

def lsc_correction(raw):
    """
    镜头阴影校正（实际项目中第三步执行）
    :param raw: 输入 16bit RAW 图（已做 BLC+DPC）
    :return: 校正后 RAW 图
    """
    h, w = raw.shape
    
    # 生成标定好的衰减系数矩阵（实际项目中是产线标定的 LUT）
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    center_y, center_x = h//2, w//2
    distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    max_distance = np.sqrt((h//2)**2 + (w//2)**2)
    lsc_attn = 1 - 0.5 * (distance / max_distance)
    
    # 避免除以 0，最小系数设为 0.1
    lsc_attn = np.maximum(lsc_attn, 0.1)
    
    # LSC 校正：原始值 / 衰减系数，恢复均匀亮度
    raw_corrected = (raw.astype(np.float32) / lsc_attn).astype(np.uint16)
    # 截断到 RAW 有效值范围
    raw_corrected = np.clip(raw_corrected, 0, RAW_MAX_VALUE)
    
    return raw_corrected

def normalize_for_display(raw):
    """
    将 16bit RAW 图归一化到 0~255，用于显示
    """
    return (raw / np.max(raw) * 255).astype(np.uint8)

# ===================== 3. 主流程执行 =====================
if __name__ == "__main__":
    # 1. 生成带故障的 RAW 图
    raw_faulty = generate_faulty_raw()
    raw_faulty_show = normalize_for_display(raw_faulty)
    
    # 2. 执行 ISP 前处理（正确顺序：BLC → DPC → LSC）
    raw_after_blc = blc_correction(raw_faulty, BLACK_LEVEL)
    raw_after_dpc = dpc_correction(raw_after_blc)
    raw_after_lsc = lsc_correction(raw_after_dpc)
    
    # 3. 归一化用于显示
    raw_blc_show = normalize_for_display(raw_after_blc)
    raw_dpc_show = normalize_for_display(raw_after_dpc)
    raw_lsc_show = normalize_for_display(raw_after_lsc)
    
    # 4. 绘制对比图
    plt.figure(figsize=(16, 12))
    
    # 原始故障图
    plt.subplot(2, 2, 1)
    plt.imshow(raw_faulty_show, cmap='gray')
    plt.title('1. 原始故障 RAW 图（含坏点+黑电平+暗角）', fontsize=12)
    plt.axis('off')
    
    # BLC 校正后
    plt.subplot(2, 2, 2)
    plt.imshow(raw_blc_show, cmap='gray')
    plt.title('2. BLC 黑电平校正后（暗部还原为0）', fontsize=12)
    plt.axis('off')
    
    # DPC 校正后
    plt.subplot(2, 2, 3)
    plt.imshow(raw_dpc_show, cmap='gray')
    plt.title('3. DPC 坏点校正后（去除白点/黑点）', fontsize=12)
    plt.axis('off')
    
    # LSC 校正后（最终结果）
    plt.subplot(2, 2, 4)
    plt.imshow(raw_lsc_show, cmap='gray')
    plt.title('4. LSC 镜头暗角校正后（亮度均匀）', fontsize=12)
    plt.axis('off')
    
    # 保存对比图
    plt.tight_layout()
    plt.savefig('isp_preprocess_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 打印关键信息
    print("=== ISP 前处理完成 ===")
    print(f"原始图平均亮度：{np.mean(raw_faulty):.2f}")
    print(f"BLC 后平均亮度：{np.mean(raw_after_blc):.2f}")
    print(f"DPC 后平均亮度：{np.mean(raw_after_dpc):.2f}")
    print(f"LSC 后平均亮度：{np.mean(raw_after_lsc):.2f}")