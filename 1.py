import cv2 as cv
#import matplotlib.pyplot as plt

print("version",cv.__version__)
img = cv.imread("data/im0.png") 
if img is None:
    print("Error:无法加载图像")
    exit()

# 2. 查看基本信息
print("图片形状 (高, 宽, 通道) =", img.shape)
height, width, channel = img.shape

print("高度 =", height)
print("宽度 =", width)
print("通道数 =", channel)

pixel = img[100,100]
print("BGR Value:",pixel)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

cv.imshow("show",img)
cv.imshow("gray",gray)
cv.waitKey(0)

