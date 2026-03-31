import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局配置 =====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 复用之前的基础函数 =====================
def generate_rggb_raw(h=256, w=256):
    raw = np.random.randint(128, 600, (h, w), dtype=np.uint16)
    for y in range(h):
        for x in range(w):
            if y % 2 == 0 and x % 2 == 0:
                raw[y, x] = np.clip(raw[y, x] * 0.9, 0, 1023)
            elif y % 2 == 0 and x % 2 == 1:
                raw[y, x] = np.clip(raw[y, x] * 1.0, 0, 1023)
            elif y % 2 == 1 and x % 2 == 0:
                raw[y, x] = np.clip(raw[y, x] * 1.0, 0, 1023)
            else:
                raw[y, x] = np.clip(raw[y, x] * 0.8, 0, 1023)
    return raw

def demosaic_bilinear(rggb):
    h, w = rggb.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
    R[0::2, 0::2] = rggb[0::2, 0::2]
    G[0::2, 1::2] = rggb[0::2, 1::2]
    G[1::2, 0::2] = rggb[1::2, 0::2]
    B[1::2, 1::2] = rggb[1::2, 1::2]

    # 插值 G
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            neighbors = []
            if y + 1 < h: neighbors.append(rggb[y + 1, x])
            if y - 1 >= 0: neighbors.append(rggb[y - 1, x])
            if x + 1 < w: neighbors.append(rggb[y, x + 1])
            if x - 1 >= 0: neighbors.append(rggb[y, x - 1])
            if neighbors: G[y, x] = np.mean(neighbors)
    for y in range(1, h, 2):
        for x in range(1, w, 2):
            neighbors = []
            if y + 1 < h: neighbors.append(rggb[y + 1, x])
            if y - 1 >= 0: neighbors.append(rggb[y - 1, x])
            if x + 1 < w: neighbors.append(rggb[y, x + 1])
            if x - 1 >= 0: neighbors.append(rggb[y, x - 1])
            if neighbors: G[y, x] = np.mean(neighbors)

    # 插值 R、B
    for y in range(h):
        for x in range(w):
            if y % 2 == 0 and x % 2 == 0: continue
            elif y % 2 == 0 and x % 2 == 1:
                r_list = [R[y, x-1]] if x-1>=0 else []
                r_list += [R[y, x+1]] if x+1<w else []
                R[y, x] = np.mean(r_list) if r_list else 0
                b_list = [B[y+1, x]] if y+1<h else []
                b_list += [B[y-1, x]] if y-1>=0 else []
                B[y, x] = np.mean(b_list) if b_list else 0
            elif y % 2 == 1 and x % 2 == 0:
                r_list = [R[y-1, x]] if y-1>=0 else []
                r_list += [R[y+1, x]] if y+1<h else []
                R[y, x] = np.mean(r_list) if r_list else 0
                b_list = [B[y, x-1]] if x-1>=0 else []
                b_list += [B[y, x+1]] if x+1<w else []
                B[y, x] = np.mean(b_list) if b_list else 0
            else:
                r_list = []
                if y-1>=0 and x-1>=0: r_list.append(R[y-1, x-1])
                if y-1>=0 and x+1<w: r_list.append(R[y-1, x+1])
                if y+1<h and x-1>=0: r_list.append(R[y+1, x-1])
                if y+1<h and x+1<w: r_list.append(R[y+1, x+1])
                R[y, x] = np.mean(r_list) if r_list else 0

    rgb[...,0], rgb[...,1], rgb[...,2] = R, G, B
    return np.clip(rgb, 0, 1023)

def awb_gray_world(rgb):
    rgb = rgb.astype(np.float32)
    avg_r, avg_g, avg_b = np.mean(rgb[...,0]), np.mean(rgb[...,1]), np.mean(rgb[...,2])
    kr, kb = avg_g/(avg_r+1e-6), avg_g/(avg_b+1e-6)
    rgb[...,0] *= kr
    rgb[...,2] *= kb
    return np.clip(rgb, 0, 1023)

def apply_ccm(rgb, ccm):
    h, w = rgb.shape[:2]
    rgb_flat = rgb.reshape(-1, 3)
    rgb_corrected = rgb_flat @ ccm.T
    return np.clip(rgb_corrected.reshape(h, w, 3), 0, 1023)

# ===================== 2. 新增模块函数（Gamma/锐化/降噪/YUV） =====================
def gamma_correction(rgb, gamma=2.2):
    rgb_norm = rgb / 1023.0
    rgb_gamma = np.power(rgb_norm, 1.0 / gamma)
    return np.clip(rgb_gamma * 1023.0, 0, 1023)

def usm_sharpening(rgb, sigma=1.0, amount=1.5):
    rgb_8bit = (rgb / 1023.0 * 255).astype(np.uint8)
    rgb_blur = cv2.GaussianBlur(rgb_8bit, (0,0), sigmaX=sigma)
    rgb_blur = (rgb_blur / 255.0 * 1023.0).astype(np.float32)
    rgb_sharp = rgb + (rgb - rgb_blur) * amount
    return np.clip(rgb_sharp, 0, 1023)

def bilateral_denoising(rgb, d=5, sigma_color=75, sigma_space=75):
    rgb_8bit = (rgb / 1023.0 * 255).astype(np.uint8)
    rgb_denoise = cv2.bilateralFilter(rgb_8bit, d, sigma_color, sigma_space)
    return (rgb_denoise / 255.0 * 1023.0).astype(np.float32)

def rgb2yuv(rgb):
    rgb_8bit = (rgb / 1023.0 * 255).astype(np.float32)
    r, g, b = rgb_8bit[...,0], rgb_8bit[...,1], rgb_8bit[...,2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128.0
    v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128.0
    yuv = np.stack([np.clip(y,0,255), np.clip(u,0,255), np.clip(v,0,255)], axis=-1).astype(np.uint8)
    h, w = yuv.shape[:2]
    u_420 = cv2.resize(u, (w//2, h//2), cv2.INTER_LINEAR)
    v_420 = cv2.resize(v, (w//2, h//2), cv2.INTER_LINEAR)
    return yuv, (y.astype(np.uint8), u_420, v_420)

# ===================== 3. 验证函数 =====================
def verify_gamma_effect(rgb_before, rgb_after):
    dark_mask = (rgb_before[...,1] >=0) & (rgb_before[...,1] <=200)
    bright_mask = (rgb_before[...,1] >=800) & (rgb_before[...,1] <=1023)
    d_mean_b = np.mean(rgb_before[dark_mask,1])
    d_mean_a = np.mean(rgb_after[dark_mask,1])
    b_mean_b = np.mean(rgb_before[bright_mask,1])
    b_mean_a = np.mean(rgb_after[bright_mask,1])
    print(f"\n=== Gamma 校正效果 ===")
    print(f"暗部亮度：{d_mean_b:.2f} → {d_mean_a:.2f} (变化：{(d_mean_a-d_mean_b)/d_mean_b*100:.2f}%)")
    print(f"亮部亮度：{b_mean_b:.2f} → {b_mean_a:.2f} (变化：{(b_mean_a-b_mean_b)/b_mean_b*100:.2f}%)")
    return d_mean_b, d_mean_a, b_mean_b, b_mean_a

def verify_sharpen_effect(rgb_before, rgb_after):
    g_b = rgb_before[...,1].astype(np.uint8)
    g_a = rgb_after[...,1].astype(np.uint8)
    grad_b = np.sqrt(cv2.Sobel(g_b, cv2.CV_64F,1,0)**2 + cv2.Sobel(g_b, cv2.CV_64F,0,1)**2)
    grad_a = np.sqrt(cv2.Sobel(g_a, cv2.CV_64F,1,0)**2 + cv2.Sobel(g_a, cv2.CV_64F,0,1)**2)
    print(f"\n=== 锐化效果 ===")
    print(f"边缘梯度：{np.mean(grad_b):.2f} → {np.mean(grad_a):.2f} (提升：{(np.mean(grad_a)-np.mean(grad_b))/np.mean(grad_b)*100:.2f}%)")
    return np.mean(grad_b), np.mean(grad_a)

def verify_denoise_effect(rgb_before, rgb_after):
    std_b = np.std(rgb_before[...,1])
    std_a = np.std(rgb_after[...,1])
    print(f"\n=== 降噪效果 ===")
    print(f"像素标准差：{std_b:.2f} → {std_a:.2f} (降低：{(std_b-std_a)/std_b*100:.2f}%)")
    return std_b, std_a

def verify_yuv_conversion(rgb, yuv):
    rgb_8bit = (rgb / 1023.0 * 255).astype(np.float32)
    y_theory = 0.299*rgb_8bit[...,0] + 0.587*rgb_8bit[...,1] + 0.114*rgb_8bit[...,2]
    y_actual = yuv[...,0].astype(np.float32)
    error = abs(np.mean(y_actual) - np.mean(y_theory)) / np.mean(y_theory) * 100
    print(f"\n=== YUV 转换效果 ===")
    print(f"Y 通道均值误差：{error:.2f}%")
    return error

# ===================== 4. 主流程 =====================
if __name__ == "__main__":
    # 生成 RAW 图
    raw = generate_rggb_raw(256, 256)
    
    # ISP 流水线全流程
    rgb_dem = demosaic_bilinear(raw)          # 去马赛克
    rgb_awb = awb_gray_world(rgb_dem)         # 白平衡
    rgb_ccm = apply_ccm(rgb_awb, np.array([[1.6,-0.4,-0.2],[-0.2,1.5,-0.3],[-0.1,-0.2,1.3]], dtype=np.float32))  # 色彩校正
    rgb_gamma = gamma_correction(rgb_ccm)     # Gamma 校正
    rgb_sharp = usm_sharpening(rgb_gamma)     # 锐化
    rgb_denoise = bilateral_denoising(rgb_sharp)  # 降噪
    yuv, (y_420, u_420, v_420) = rgb2yuv(rgb_denoise)  # YUV 变换

    # 效果验证
    verify_gamma_effect(rgb_ccm, rgb_gamma)
    verify_sharpen_effect(rgb_gamma, rgb_sharp)
    verify_denoise_effect(rgb_sharp, rgb_denoise)
    verify_yuv_conversion(rgb_denoise, yuv)

    # 可视化全流程
    plt.figure(figsize=(20, 8))
    # 1. 原始 RAW
    plt.subplot(2, 4, 1)
    plt.imshow(raw, cmap='gray')
    plt.title("1. 原始 Bayer RAW")
    plt.axis('off')
    # 2. Demosaic+AWB+CCM
    plt.subplot(2, 4, 2)
    plt.imshow((rgb_ccm/1023*255).astype(np.uint8))
    plt.title("2. Demosaic+AWB+CCM")
    plt.axis('off')
    # 3. Gamma 校正
    plt.subplot(2, 4, 3)
    plt.imshow((rgb_gamma/1023*255).astype(np.uint8))
    plt.title("3. Gamma 校正后")
    plt.axis('off')
    # 4. 锐化后
    plt.subplot(2, 4, 4)
    plt.imshow((rgb_sharp/1023*255).astype(np.uint8))
    plt.title("4. USM 锐化后")
    plt.axis('off')
    # 5. 降噪后
    plt.subplot(2, 4, 5)
    plt.imshow((rgb_denoise/1023*255).astype(np.uint8))
    plt.title("5. 双边滤波降噪后")
    plt.axis('off')
    # 6. YUV Y 通道
    plt.subplot(2, 4, 6)
    plt.imshow(yuv[...,0], cmap='gray')
    plt.title("6. YUV Y 通道（亮度）")
    plt.axis('off')
    # 7. YUV U 通道
    plt.subplot(2, 4, 7)
    plt.imshow(yuv[...,1], cmap='gray')
    plt.title("7. YUV U 通道（色度）")
    plt.axis('off')
    # 8. YUV V 通道
    plt.subplot(2, 4, 8)
    plt.imshow(yuv[...,2], cmap='gray')
    plt.title("8. YUV V 通道（色度）")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 打印 YUV420 尺寸信息
    print(f"\n=== YUV420 尺寸信息 ===")
    print(f"Y 通道尺寸：{y_420.shape}")
    print(f"U 通道尺寸（下采样）：{u_420.shape}")
    print(f"V 通道尺寸（下采样）：{v_420.shape}")