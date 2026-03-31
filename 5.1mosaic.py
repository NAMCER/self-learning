import cv2
import numpy as np

#Creat Bayer raw
img = cv2.imread("D:/workspace/vscode/data/im0.png")
img_rgb =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
height, width, channel = img.shape
height_, width_, channel_ = img_rgb.shape

b,g,r =cv2.split(img)
raw = np.zeros((height,width),dtype=np.uint8)
raw[0::2, 0::2] = r[0::2, 0::2] # R
raw[0::2, 1::2] = g[0::2, 1::2] # G1
raw[1::2, 0::2] = g[1::2, 0::2] # G2
raw[1::2, 1::2] = b[1::2, 1::2] # B
raw.tofile('im0_bgr_2724x1848_rggb_8bit.raw')
print(f"RAW文件已生成: {width}x{height}, 格式: RGGB, 大小: {width*height} 字节")

b_,g_,r_ =cv2.split(img_rgb)
raw_= np.zeros((height_,width_),dtype=np.uint8)
raw_[0::2, 0::2] = r_[0::2, 0::2] # R
raw_[0::2, 1::2] = g_[0::2, 1::2] # G1
raw_[1::2, 0::2] = g_[1::2, 0::2] # G2
raw_[1::2, 1::2] = b_[1::2, 1::2] # B

raw_.tofile('im0_rgb_2724x1848_rggb_8bit.raw')
print(f"RAW文件已生成: {width_}x{height_}, 格式: RGGB, 大小: {width_*height_} 字节")



# #可视化
# raw_data = np.fromfile(r'D:\workspace\vscode\test_2724x1848_rggb_8bit.raw', dtype=np.uint8)
# raw_img = raw_data.reshape((1848, 2724))
# cv2.imshow('Raw Gray', cv2.resize(raw_img, (960, 640))) # 缩放一下方便观察
# #color_preview = cv2.cvtColor(raw_img, cv2.COLOR_BAYER_BG2BGR)
# color_preview = cv2.cvtColor(raw_img, cv2.COLOR_BAYER_RGGB2RGB)
# cv2.imshow('Demosaic Preview', cv2.resize(color_preview, (960, 640)))
# cv2.waitKey(0)
