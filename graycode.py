import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_synthetic_depth_map(width, height):
    depth_map = np.ones((height, width), dtype=np.float32) * 1000.0
    cx, cy = width // 2, height // 2
    radius = 200
    y, x = np.ogrid[:height, :width]
    dist_sq = (x - cx)**2 + (y - cy)**2
    mask = dist_sq <= radius**2
    z_center = 800.0
    depth_map[mask] = z_center - np.sqrt(radius**2 - dist_sq[mask])
    return depth_map

def simulate_gray_projection(depth_map, width, height, N=10):
    f, B = 800.0, 100.0
    proj_width = 1024 
    
    u_cam, _ = np.meshgrid(np.arange(width), np.arange(height))
    # 计算理论上的投影仪坐标: u_proj = u_cam - f*B/Z
    disparity = (f * B) / depth_map
    u_proj = u_cam - disparity
    u_proj_idx = np.round(u_proj).astype(np.int32)
    
    # 有效投影范围掩模
    mask = (u_proj_idx >= 0) & (u_proj_idx < proj_width)
    
    # 十进制转格雷码: G = B ^ (B >> 1)
    gray_values = np.zeros_like(u_proj_idx)
    gray_values[mask] = u_proj_idx[mask] ^ (u_proj_idx[mask] >> 1)
    
    pattern_images = np.zeros((N, height, width), dtype=np.uint8)
    for i in range(N):
        bit_pos = (N - 1) - i 
        bits = (gray_values >> bit_pos) & 1
        pattern_images[i] = (bits * 255).astype(np.uint8)
        
    return pattern_images, mask

def decode_gray_reconstruct(pattern_images, width, height, N=10):
    # 1. 恢复格雷码数值
    gray_decoded = np.zeros((height, width), dtype=np.int32)
    for i in range(N):
        bit_pos = (N - 1) - i
        bit_val = (pattern_images[i] > 127).astype(np.int32)
        gray_decoded |= (bit_val << bit_pos)
        
    # 2. 格雷码转二进制 (串行异或还原)
    binary_decoded = gray_decoded.copy()
    for i in range(1, N):
        binary_decoded ^= (gray_decoded >> i)
        # 注意：这里不能简单用上面那个简写，正确逻辑如下：
    
    # 修正格雷码转换逻辑
    res = np.zeros_like(gray_decoded)
    for i in range(N):
        res ^= (gray_decoded >> i)
    # 这里的 res 就是 u_proj_decoded
    u_proj_decoded = res
    
    # 3. 重建
    f, B = 800.0, 100.0
    u_cam, _ = np.meshgrid(np.arange(width), np.arange(height))
    disp = u_cam.astype(np.float32) - u_proj_decoded.astype(np.float32)
    
    # 过滤掉不合理的视差（避免分母为0或负数）
    valid_mask = disp > 0
    Z_reconstructed = np.ones((height, width)) * 1000.0 # 背景默认1000
    Z_reconstructed[valid_mask] = (f * B) / disp[valid_mask]
    
    return Z_reconstructed, u_proj_decoded

# --- 运行 ---
width, height, N = 640, 480, 10
gt_depth = generate_synthetic_depth_map(width, height)
patterns, mask = simulate_gray_projection(gt_depth, width, height, N)
rec_depth, rec_code = decode_gray_reconstruct(patterns, width, height, N)

# --- 画图 ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(rec_depth, vmin=600, vmax=1000)
plt.title("Corrected Reconstructed")
plt.subplot(1, 2, 2)
error = np.abs(gt_depth - rec_depth)
error[gt_depth >= 1000] = 0 # 忽略背景误差
plt.imshow(error, vmin=0, vmax=5, cmap='hot')
plt.title("Corrected Error (mm)")
plt.show()