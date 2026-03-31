import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 生成模拟 RGGB RAW 图 =====================
def generate_rggb_raw(h=256, w=256):
    raw = np.random.randint(128, 600, (h, w), dtype=np.uint16)

    # RGGB 模拟
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

# ===================== 2. Demosaic 去马赛克（双线性） =====================
def demosaic_bilinear(rggb):
    h, w = rggb.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]

    # 赋值已知通道
    R[0::2, 0::2] = rggb[0::2, 0::2]
    G[0::2, 1::2] = rggb[0::2, 1::2]
    G[1::2, 0::2] = rggb[1::2, 0::2]
    B[1::2, 1::2] = rggb[1::2, 1::2]

    # 双线性插值 G
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            neighbors = []
            if y + 1 < h: neighbors.append(rggb[y + 1, x])
            if y - 1 >= 0: neighbors.append(rggb[y - 1, x])
            if x + 1 < w: neighbors.append(rggb[y, x + 1])
            if x - 1 >= 0: neighbors.append(rggb[y, x - 1])
            if neighbors:
                G[y, x] = np.mean(neighbors)

    for y in range(1, h, 2):
        for x in range(1, w, 2):
            neighbors = []
            if y + 1 < h: neighbors.append(rggb[y + 1, x])
            if y - 1 >= 0: neighbors.append(rggb[y - 1, x])
            if x + 1 < w: neighbors.append(rggb[y, x + 1])
            if x - 1 >= 0: neighbors.append(rggb[y, x - 1])
            if neighbors:
                G[y, x] = np.mean(neighbors)

    # 插值 R、B
    for y in range(h):
        for x in range(w):
            if y % 2 == 0 and x % 2 == 0:
                continue
            elif y % 2 == 0 and x % 2 == 1:
                # G位置插值R、B
                r_list = []
                if x - 1 >= 0: r_list.append(R[y, x - 1])
                if x + 1 < w: r_list.append(R[y, x + 1])
                R[y, x] = np.mean(r_list) if r_list else R[y, x - 1]

                b_list = []
                if y + 1 < h: b_list.append(B[y + 1, x])
                if y - 1 >= 0: b_list.append(B[y - 1, x])
                B[y, x] = np.mean(b_list) if b_list else B[y + 1, x]

            elif y % 2 == 1 and x % 2 == 0:
                # G位置插值R、B
                r_list = []
                if y - 1 >= 0: r_list.append(R[y - 1, x])
                if y + 1 < h: r_list.append(R[y + 1, x])
                R[y, x] = np.mean(r_list) if r_list else R[y - 1, x]

                b_list = []
                if x - 1 >= 0: b_list.append(B[y, x - 1])
                if x + 1 < w: b_list.append(B[y, x + 1])
                B[y, x] = np.mean(b_list) if b_list else B[y, x - 1]

            else:
                # B位置插值R
                r_list = []
                if y - 1 >= 0 and x - 1 >= 0: r_list.append(R[y - 1, x - 1])
                if y - 1 >= 0 and x + 1 < w: r_list.append(R[y - 1, x + 1])
                if y + 1 < h and x - 1 >= 0: r_list.append(R[y + 1, x - 1])
                if y + 1 < h and x + 1 < w: r_list.append(R[y + 1, x + 1])
                R[y, x] = np.mean(r_list) if r_list else 0

    rgb[..., 0] = R
    rgb[..., 1] = G
    rgb[..., 2] = B
    rgb = np.clip(rgb, 0, 1023)
    return rgb

# ===================== 3. AWB 自动白平衡（Gray World 经典算法） =====================
def awb_gray_world(rgb):
    rgb = rgb.astype(np.float32)
    avg_r = np.mean(rgb[..., 0])
    avg_g = np.mean(rgb[..., 1])
    avg_b = np.mean(rgb[..., 2])

    kr = avg_g / (avg_r + 1e-6)
    kb = avg_g / (avg_b + 1e-6)

    rgb[..., 0] *= kr
    rgb[..., 2] *= kb
    rgb = np.clip(rgb, 0, 1023)
    return rgb

# ===================== 4. CCM 色彩校正矩阵 =====================
def apply_ccm(rgb, ccm):
    h, w, _ = rgb.shape
    rgb_flat = rgb.reshape(-1, 3)
    rgb_corrected = rgb_flat @ ccm.T
    rgb_corrected = rgb_corrected.reshape(h, w, 3)
    rgb_corrected = np.clip(rgb_corrected, 0, 1023)
    return rgb_corrected

# 标准 sRGB 校正 CCM（常用）
ccm_standard = np.array([
    [1.6, -0.4, -0.2],
    [-0.2, 1.5, -0.3],
    [-0.1, -0.2, 1.3]
], dtype=np.float32)

# ===================== 工具：显示 =====================
def show(img, title):
    disp = (img / np.max(img) * 255).astype(np.uint8)
    plt.imshow(disp)
    plt.title(title)
    plt.axis('off')

# ===================== 主流程 =====================
if __name__ == "__main__":
    raw = generate_rggb_raw(256, 256)

    # 1. Demosaic
    rgb_dem = demosaic_bilinear(raw)

    # 2. AWB
    rgb_awb = awb_gray_world(rgb_dem)

    # 3. CCM
    rgb_ccm = apply_ccm(rgb_awb, ccm_standard)

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(raw, cmap='gray')
    plt.title("原始 Bayer RAW")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    show(rgb_dem, "Demosaic 后")

    plt.subplot(1, 4, 3)
    show(rgb_awb, "AWB 白平衡后")

    plt.subplot(1, 4, 4)
    show(rgb_ccm, "CCM 色彩校正后")

    plt.tight_layout()
    plt.show()