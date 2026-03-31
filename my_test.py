# import open3d as o3d
# import numpy as np

# print("->正在加载点云... ")
# pcd1 = o3d.io.read_point_cloud("weighted_fused_cloud.pcd")
# print(pcd1)

# #print("->正在保存点云")
# #o3d.io.write_point_cloud("write.pcd", pcd, write_ascii=False)	# 默认false，保存为Binarty；True 保存为ASICC形式
# #print(pcd)

# #pcd2 = o3d.io.read_point_cloud("write.pcd")
# #print(pcd2)

# print("->可视化点云")
# o3d.visualization.draw_geometries([pcd1])

import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_synthetic_depth_map(width, height):
    """
    生成一个虚拟的深度图（场景中有一个球体）
    :return: depth_map (Z值)
    """
    # 初始化背景深度为 1000mm
    depth_map = np.ones((height, width), dtype=np.float32) * 1000.0
    
    # 在中心生成一个球体
    cx, cy = width // 2, height // 2
    radius = 200
    y, x = np.ogrid[:height, :width]
    dist_sq = (x - cx)**2 + (y - cy)**2
    mask = dist_sq <= radius**2
    
    # 球面方程 Z = Z_center - sqrt(R^2 - r^2)
    # 假设球心距离相机 800mm
    z_center = 800.0
    depth_map[mask] = z_center - np.sqrt(radius**2 - dist_sq[mask])
    
    return depth_map

def simulate_projection(depth_map, width, height, N=10):
    """
    模拟 N=10 的二进制编码投影过程
    """
    # 系统参数 (简单的立体视觉模型)
    f = 800.0       # 焦距
    B = 100.0       # 基线距离 (Camera-Projector Baseline)
    proj_width = 1024 # 投影仪宽度 2^10 = 1024
    
    # 存储生成的10幅图案
    pattern_images = np.zeros((N, height, width), dtype=np.uint8)
    
    # 遍历每一个像素 (向量化计算加速)
    # 1. 获取相机像素坐标
    u_cam, v_cam = np.meshgrid(np.arange(width), np.arange(height))
    Z = depth_map
    
    # 2. 理想三角测量公式逆推: 算出该点对应的投影仪列坐标 u_proj
    # 公式: Z = (f * B) / (u_cam - u_proj)  --> u_proj = u_cam - (f * B) / Z
    # 注意：这只是一个简化模型，实际中需要用到旋转平移矩阵
    # 为了模拟视差：disparity = u_cam - u_proj = f * B / Z
    disparity = (f * B) / Z
    u_proj = u_cam - disparity
    
    # 将投影仪坐标转换为整数索引 (0 ~ 1023)
    u_proj_idx = np.round(u_proj).astype(np.int32)
    
    # 过滤掉超出投影仪视场的点
    mask = (u_proj_idx >= 0) & (u_proj_idx < proj_width)
    
    # 3. 生成二进制图案
    for i in range(N):
        # 提取第 i 位的比特值 (0 或 1)
        # N=10, i=0 是最低位(LSB), i=9 是最高位(MSB) 
        # 这里我们按位操作: (index >> bit_pos) & 1
        # 通常 Pattern 1 是 MSB 还是 LSB 取决于定义，这里假设 Pattern 0 是 MSB
        bit_pos = (N - 1) - i 
        
        bits = (u_proj_idx >> bit_pos) & 1
        
        # 生成图像: 1 -> 255(白), 0 -> 0(黑)
        img = np.zeros_like(bits, dtype=np.uint8)
        img[mask] = bits[mask] * 255
        pattern_images[i] = img
        
    return pattern_images, u_proj

def decode_and_reconstruct(pattern_images, width, height, N=10):
    """
    解码二进制图案并重建点云(深度)
    """
    # 1. 解码 (Decoding)
    # 将10张图合并回一个整数索引值
    u_proj_decoded = np.zeros((height, width), dtype=np.int32)
    
    for i in range(N):
        bit_pos = (N - 1) - i
        # 阈值化：大于127算1，否则算0
        bit_val = (pattern_images[i] > 127).astype(np.int32)
        u_proj_decoded += bit_val << bit_pos
        
    # 2. 三角测量重建 (Triangulation Reconstruction)
    # 公式: Z = (f * B) / (u_cam - u_proj)
    f = 800.0
    B = 100.0
    
    u_cam, _ = np.meshgrid(np.arange(width), np.arange(height))
    
    # 计算视差 disparity
    disparity = u_cam.astype(np.float32) - u_proj_decoded.astype(np.float32)
    
    # 避免除以0
    disparity[disparity == 0] = 0.001
    
    # 重建深度 Z
    Z_reconstructed = (f * B) / disparity
    
    return Z_reconstructed, u_proj_decoded

# --- 主程序 ---
width, height = 640, 480
N = 10

# 1. 生成地面真值 (Ground Truth)
gt_depth = generate_synthetic_depth_map(width, height)

# 2. 模拟投影与拍摄过程 (生成10张条纹图)
patterns, gt_u_proj = simulate_projection(gt_depth, width, height, N)

# 3. 算法重建
rec_depth, rec_code = decode_and_reconstruct(patterns, width, height, N)

# --- 可视化结果 ---
plt.figure(figsize=(15, 8))

# 显示第5幅投射图案 (模拟相机拍到的样子)
plt.subplot(2, 3, 1)
plt.imshow(patterns[5], cmap='gray')
plt.title(f'Captured Pattern #{5} (Bit 4)')
plt.axis('off')

# 显示第9幅投射图案 (LSB, 最细的条纹)
plt.subplot(2, 3, 2)
plt.imshow(patterns[9], cmap='gray')
plt.title(f'Captured Pattern #{9} (LSB)')
plt.axis('off')

# 显示解码出来的投影仪坐标 (u_proj)
plt.subplot(2, 3, 3)
plt.imshow(rec_code, cmap='jet')
plt.title('Decoded Projector Column Index')
plt.axis('off')

# 显示原始深度图 (Ground Truth)
plt.subplot(2, 3, 4)
plt.imshow(gt_depth, cmap='viridis', vmin=600, vmax=1000)
plt.title('Ground Truth Depth')
plt.axis('off')

# 显示重建深度图 (Reconstructed)
plt.subplot(2, 3, 5)
plt.imshow(rec_depth, cmap='viridis', vmin=600, vmax=1000)
plt.title('Reconstructed Depth Cloud')
plt.axis('off')

# 显示误差 (Error)
error = np.abs(gt_depth - rec_depth)
error[rec_depth > 2000] = 0 # 忽略背景噪点
plt.subplot(2, 3, 6)
plt.imshow(error, cmap='hot', vmin=0, vmax=10)
plt.title('Reconstruction Error (mm)')
plt.axis('off')

plt.tight_layout()
plt.show()

