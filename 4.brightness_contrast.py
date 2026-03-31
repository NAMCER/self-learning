import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 解决matplotlib中文显示问题
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False

#读取并转换RGB、转换浮点型避免整数溢出
img = cv2.imread("data/im0.png")
# 新增：校验图片是否读取成功（避免路径错误导致的隐性问题）
if img is None:
    raise ValueError("图片读取失败！请检查路径是否正确，比如改为绝对路径：C:/data/im0.png")

img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_float = img_rgb.astype(np.float32)

#亮度/对比度（线性）
def adjust_brightness_contrast_linear(img,brightness =0,contrast=1.0):
    """
    线性调整亮度和对比度
    :param img: 输入图像（RGB，浮点型）
    :param brightness: 亮度偏移量（-255 ~ 255），正数调亮，负数调暗
    :param contrast: 对比度系数（0.0 ~ 3.0），>1提高，<1降低
    :return: 调整后的图像（uint8）
    """
    #先对比度再亮度
    img_contrast= (img -128)*contrast +128
    img_bright = img_contrast + brightness
    # 截断：保证像素值在0~255之间（关键！）
    img_bright = np.clip(img_bright,0,255)

    return img_bright.astype(np.uint8)

# 调亮：亮度+50，对比度不变
img_bright = adjust_brightness_contrast_linear(img_float, brightness=50, contrast=1.0)
# 调暗：亮度-50，对比度不变
img_dark = adjust_brightness_contrast_linear(img_float, brightness=-50, contrast=1.0)
# 提高对比度+调亮
img_high_contrast = adjust_brightness_contrast_linear(img_float, brightness=20, contrast=1.5)

#伽马变换调亮度（ISP 实际用）
def adjust_brightness_gamma(img, gamma = 1.0):
    """
    伽马变换调亮度（ISP 常用，非线性调整，更符合人眼感知）
    :param img: 输入图像（RGB，uint8）
    :param gamma: 伽马值，<1调亮，>1调暗
    :return: 调整后的图像
    """
    # 构建 Gamma 查找表（核心：避免逐像素计算，提高效率）
    # 修正注释：和代码逻辑一致
    inv_gamma = 1.0 / gamma  # 这一步是为了让 "gamma值" 符合直观认知
    gamma_table = np.array([((i / 255.0) ** gamma) * 255 
                            for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(img,gamma_table)

# 伽马调亮（gamma=0.5）
img_gamma_bright = adjust_brightness_gamma(img_rgb, gamma=0.5)
# 伽马调暗（gamma=2.0）
img_gamma_dark = adjust_brightness_gamma(img_rgb, gamma=2.0)

#显示
plt.figure(figsize=(15, 10))

# 子图1：原图
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("原图")
plt.axis("off")

# 子图2：线性调亮
plt.subplot(2, 3, 2)
plt.imshow(img_bright)
plt.title("线性调亮（+50）")
plt.axis("off")

# 子图3：线性调暗
plt.subplot(2, 3, 3)
plt.imshow(img_dark)
plt.title("线性调暗（-50）")
plt.axis("off")

# 子图4：提高对比度+调亮
plt.subplot(2, 3, 4)
plt.imshow(img_high_contrast)
plt.title("提高对比度（1.5）+ 调亮（+20）")
plt.axis("off")

# 子图5：伽马调亮
plt.subplot(2, 3, 5)
plt.imshow(img_gamma_bright)
plt.title("伽马调亮（gamma=0.5）")
plt.axis("off")

# 子图6：伽马调暗
plt.subplot(2, 3, 6)
plt.imshow(img_gamma_dark)
plt.title("伽马调暗（gamma=2.0）")
plt.axis("off")

# 保存结果
plt.savefig("brightness_contrast_result.png", dpi=150)
plt.show()

# 新增：打印所有图像的平均亮度（全面验证）
def get_avg_brightness(img):
    return np.mean(img)

print("=== 亮度统计（平均像素值）===")
print("原图平均亮度：", round(get_avg_brightness(img_rgb), 2))
print("线性调亮后：", round(get_avg_brightness(img_bright), 2))
print("线性调暗后：", round(get_avg_brightness(img_dark), 2))
print("gamma=0.5 平均亮度：", round(get_avg_brightness(img_gamma_bright), 2))
print("gamma=2.0 平均亮度：", round(get_avg_brightness(img_gamma_dark), 2))