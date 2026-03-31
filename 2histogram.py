import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("data/im0.png")
img_rgb =cv.cvtColor(img,cv.COLOR_BGR2RGB)
img_r = img_rgb[:,:,0]
img_g = img_rgb[:,:,1]
img_b = img_rgb[:,:,2]

def calculate_his(channel):
    hist=np.zeros(256,dtype=np.int32)
    height,width = channel.shape
    for y in range(height):
        for x in range(width):
            pixel_val =channel[y,x]
            hist[pixel_val]+=1
    return hist

def calculate_his_cv(channel):
    hist=cv.calcHist([channel],[0],None,[256],[0,256])
    return hist

# r_hist= calculate_his(img_r)
# g_hist =calculate_his(img_g)
# b_hist = calculate_his(img_b)

r_hist =calculate_his_cv(img_r)
g_hist =calculate_his_cv(img_g)
b_hist =calculate_his_cv(img_b)

#show
plt.figure(figsize=(12, 8))
plt.subplot(2,1,1)
plt.imshow(img_rgb)
plt.title("Orriginal Image")
plt.axis("off")

plt.subplot(2,1,2)
plt.plot(r_hist, color='red', label='Red')
plt.plot(g_hist, color='green',label='Green')
plt.plot(b_hist,color ='blue',label='Blue')

print("===直方图统计信息===")
# 计算每个通道的平均像素值（反映亮度）
r_mean = np.mean(img_r)
g_mean = np.mean(img_g)
b_mean = np.mean(img_b)
print(f"R 通道平均亮度：{r_mean:.2f}")
print(f"G 通道平均亮度：{g_mean:.2f}")
print(f"B 通道平均亮度：{b_mean:.2f}")
plt.show()

# 判断是否过曝/欠曝（简单版）
