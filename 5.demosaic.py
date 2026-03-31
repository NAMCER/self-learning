import cv2
import numpy as np

# #Creat Bayer raw
# img = cv2.imread("data/im0.png")
# height, width, channel = img.shape
# b,g,r =cv2.split(img)

# raw = np.zeros((height,width),dtype=np.uint8)

# raw[0::2, 0::2] = r[0::2, 0::2] # R
# raw[0::2, 1::2] = g[0::2, 1::2] # G1
# raw[1::2, 0::2] = g[1::2, 0::2] # G2
# raw[1::2, 1::2] = b[1::2, 1::2] # B

# raw.tofile('test_2724x1848_rggb_8bit.raw')
# print(f"RAW文件已生成: {width}x{height}, 格式: RGGB, 大小: {width*height} 字节")

##可视化
#raw_data = np.fromfile('test_2724x1848_rggb_8bit.raw', dtype=np.uint8)
#raw_img = raw_data.reshape((height, width))
# cv2.imshow('Raw Gray', cv2.resize(raw_img, (960, 640))) # 缩放一下方便观察
# color_preview = cv2.cvtColor(raw_img, cv2.COLOR_BAYER_BG2BGR)
# cv2.imshow('Demosaic Preview', cv2.resize(color_preview, (960, 640)))
# cv2.waitKey(0)

##Demosaic :Nearest Neighbor
def demosaic_nearest(raw,h,w):
    dst =np.zeros((h,w,3),dtype=np.uint8)
    # 偶数行、偶数列: R
    r_val = raw[0::2, 0::2]
    # 偶数行、奇数列: G1
    g1_val = raw[0::2, 1::2]
    # 奇数行、偶数列: G2
    g2_val = raw[1::2, 0::2]
    # 奇数行、奇数列: B
    b_val = raw[1::2, 1::2]
    # 3. 计算均值 G (对应 C++ 中的 uchar G = (G1 + G2) / 2)
    g_avg = ((g1_val.astype(np.uint16) + g2_val.astype(np.uint16)) // 2).astype(np.uint8)
    # 蓝色通道 (Index 0)
    dst[0::2, 0::2, 0] = b_val; dst[0::2, 1::2, 0] = b_val
    dst[1::2, 0::2, 0] = b_val; dst[1::2, 1::2, 0] = b_val
    
    # 绿色通道 (Index 1)
    dst[0::2, 0::2, 1] = g_avg; dst[0::2, 1::2, 1] = g_avg
    dst[1::2, 0::2, 1] = g_avg; dst[1::2, 1::2, 1] = g_avg
    
    # 红色通道 (Index 2)
    dst[0::2, 0::2, 2] = r_val; dst[0::2, 1::2, 2] = r_val
    dst[1::2, 0::2, 2] = r_val; dst[1::2, 1::2, 2] = r_val
    return dst

##Demosaic :双线性插值（Bilinear）
def demosaic_bilinear(raw,h,w):
    # 为了处理边界，我们对原始图进行 padding (边缘填充)
    raw_pad = np.pad(raw, (1, 1), mode='reflect').astype(np.float32)
    # 初始化输出图像 (BGR)
    dst = np.zeros((h, w, 3), dtype=np.uint8)
    # 1. 提取原始分量 (在 pad 后的图中坐标会整体偏移 +1)
    r = raw_pad[1:-1:2, 1:-1:2]
    g1 = raw_pad[1:-1:2, 2::2]
    g2 = raw_pad[2::2, 1:-1:2]
    b = raw_pad[2::2, 2::2]

    # --- 处理 G 通道 (Green) ---
    # 在 R 位置插值 G: 取四周 G1, G2 的均值
    # r_pos 对应的邻居: 上(g1_up), 下(g1_down), 左(g2_left), 右(g2_right)
    g_at_r = (raw_pad[0:-2:2, 1:-1:2] + raw_pad[2::2, 1:-1:2] + 
              raw_pad[1:-1:2, 0:-2:2] + raw_pad[1:-1:2, 2::2]) / 4
              
    # 在 B 位置插值 G: 同理
    g_at_b = (raw_pad[1:-1:2, 2::2] + raw_pad[3::2, 2::2] + 
              raw_pad[2::2, 1:-1:2] + raw_pad[2::2, 3::2]) / 4

    # --- 处理 R 通道 (Red) ---
    # 在 G1 位置插值 R: 左右两个 R 的均值
    r_at_g1 = (raw_pad[1:-1:2, 1:-1:2] + raw_pad[1:-1:2, 3::2]) / 2
    # 在 G2 位置插值 R: 上下两个 R 的均值
    r_at_g2 = (raw_pad[1:-1:2, 1:-1:2] + raw_pad[3::2, 1:-1:2]) / 2
    # 在 B 位置插值 R: 四个斜角 R 的均值
    r_at_b = (raw_pad[1:-1:2, 1:-1:2] + raw_pad[1:-1:2, 3::2] + 
              raw_pad[3::2, 1:-1:2] + raw_pad[3::2, 3::2]) / 4

    # --- 处理 B 通道 (Blue) ---
    # 在 G1 位置插值 B: 上下两个 B 的均值
    b_at_g1 = (raw_pad[0:-2:2, 2::2] + raw_pad[2::2, 2::2]) / 2
    # 在 G2 位置插值 B: 左右两个 B 的均值
    b_at_g2 = (raw_pad[2::2, 0:-2:2] + raw_pad[2::2, 2::2]) / 2
    # 在 R 位置插值 B: 四个斜角 B 的均值
    b_at_r = (raw_pad[0:-2:2, 0:-2:2] + raw_pad[0:-2:2, 2::2] + 
              raw_pad[2::2, 0:-2:2] + raw_pad[2::2, 2::2]) / 4
    
    # 填充结果 (OpenCV BGR 顺序)
    # R 位置 (Index 2=R, 1=G, 0=B)
    dst[0::2, 0::2, 2] = r
    dst[0::2, 0::2, 1] = g_at_r
    dst[0::2, 0::2, 0] = b_at_r

    # G1 位置
    dst[0::2, 1::2, 2] = r_at_g1
    dst[0::2, 1::2, 1] = g1
    dst[0::2, 1::2, 0] = b_at_g1

    # G2 位置
    dst[1::2, 0::2, 2] = r_at_g2
    dst[1::2, 0::2, 1] = g2
    dst[1::2, 0::2, 0] = b_at_g2

    # B 位置
    dst[1::2, 1::2, 2] = r_at_b
    dst[1::2, 1::2, 1] = g_at_b
    dst[1::2, 1::2, 0] = b
    return dst.astype(np.uint8)


##边缘自适应插值
def demosaic_edge_aware(raw, h, w):
    pad = 2
    raw_p = np.pad(raw, (pad, pad), mode='reflect').astype(np.float32)
    g_full = np.zeros((h + 2*pad, w + 2*pad), dtype=np.float32)
    g_full[pad:-pad, pad:-pad] = raw # 先填入原始 Bayer 中的 G 值点

    # --- 1. 插值 G 通道 (Hamilton-Adams 核心逻辑) ---
    y, x = np.meshgrid(np.arange(pad, h+pad, 2), np.arange(pad, w+pad, 2), indexing='ij')
    
    def interpolate_g_at(y_idx, x_idx):
        dh = np.abs(raw_p[y_idx, x_idx-1] - raw_p[y_idx, x_idx+1]) + \
             np.abs(2 * raw_p[y_idx, x_idx] - raw_p[y_idx, x_idx-2] - raw_p[y_idx, x_idx+2])
        dv = np.abs(raw_p[y_idx-1, x_idx] - raw_p[y_idx+1, x_idx]) + \
             np.abs(2 * raw_p[y_idx, x_idx] - raw_p[y_idx-2, x_idx] - raw_p[y_idx+2, x_idx])
        
        g_h = (raw_p[y_idx, x_idx-1] + raw_p[y_idx, x_idx+1]) / 2 + \
              (2 * raw_p[y_idx, x_idx] - raw_p[y_idx, x_idx-2] - raw_p[y_idx, x_idx+2]) / 4
        g_v = (raw_p[y_idx-1, x_idx] + raw_p[y_idx+1, x_idx]) / 2 + \
              (2 * raw_p[y_idx, x_idx] - raw_p[y_idx-2, x_idx] - raw_p[y_idx+2, x_idx]) / 4
        return np.where(dh < dv, g_h, np.where(dv < dh, g_v, (g_h + g_v) / 2))

    # 填充 R/B 位置缺失的 G
    g_full[pad:-pad:2, pad:-pad:2] = interpolate_g_at(y, x)      # R 点处的 G
    g_full[pad+1:-pad:2, pad+1:-pad:2] = interpolate_g_at(y+1, x+1)  # B 点处的 G

    # --- 2. 插值 R 和 B 通道 (色差法 Color Difference) ---
    # 我们创建一个完整的 R 图像和 B 图像
    r_full = np.zeros_like(g_full)
    b_full = np.zeros_like(g_full)

    # 计算已知点的色差 (R-G 和 B-G)
    diff_r = np.zeros_like(g_full)
    diff_b = np.zeros_like(g_full)
    diff_r[pad:-pad:2, pad:-pad:2] = raw_p[pad:-pad:2, pad:-pad:2] - g_full[pad:-pad:2, pad:-pad:2]
    diff_b[pad+1:-pad:2, pad+1:-pad:2] = raw_p[pad+1:-pad:2, pad+1:-pad:2] - g_full[pad+1:-pad:2, pad+1:-pad:2]

    # 对色差层进行简单的双线性插值 (色差层通常比较平滑，不需要梯度检测)
    def bilinear_diff(diff_layer):
        # 这是一个简单的均值模糊/插值，用来填满差值图
        kernel = np.array([[1/4, 1/2, 1/4], [1/2, 1, 1/2], [1/4, 1/2, 1/4]], dtype=np.float32)
        return cv2.filter2D(diff_layer, -1, kernel)

    full_diff_r = bilinear_diff(diff_r)
    full_diff_b = bilinear_diff(diff_b)

    # 还原彩色：R = G + diff_R, B = G + diff_B
    r_final = g_full + full_diff_r
    b_final = g_full + full_diff_b

    # --- 3. 合并并修剪边缘 ---
    dst = np.zeros((h, w, 3), dtype=np.uint8)
    dst[:, :, 2] = np.clip(r_final[pad:-pad, pad:-pad], 0, 255) # R
    dst[:, :, 1] = np.clip(g_full[pad:-pad, pad:-pad], 0, 255)  # G
    dst[:, :, 0] = np.clip(b_final[pad:-pad, pad:-pad], 0, 255) # B
    
    return dst

height, width ={1848,2724}
#可视化
raw_data = np.fromfile('test_2724x1848_rggb_8bit.raw', dtype=np.uint8).reshape(height, width)
result = demosaic_nearest(raw_data,height,width)
result1 = demosaic_bilinear(raw_data,height,width)
result2 = demosaic_edge_aware(raw_data,height,width)
cv2.imshow('Nearest Result', result)
cv2.imshow("Bilinear",result1)
cv2.imshow("edge_aware",result2)
cv2.imwrite("demosaic_nearest.png",result)
cv2.imwrite("demosaic_bilinear.png",result1)
cv2.imwrite("demosaic_edge_aware.png",result2)
cv2.waitKey(0)

