import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# --- 1. 数据集类 ---
class DemosaicDataset(Dataset):
    def __init__(self, img_dir, patch_size=128, patches_per_image=20):
        # 路径检查
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"找不到文件夹: {img_dir}")

        img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image

        if len(img_paths) == 0:
            print("警告: 文件夹内没有图片！")

        # 一次性加载所有图片到内存，避免每次读盘
        print(f"正在加载 {len(img_paths)} 张图片到内存...")
        self.images = [cv2.imread(p) for p in img_paths]
        print("图片加载完成。")

    def __len__(self):
        return len(self.images) * self.patches_per_image

    def __getitem__(self, idx):
        img = self.images[idx % len(self.images)]
        h, w, _ = img.shape
        
        # 【修正点】确保裁剪起始点是偶数，防止 Bayer 阵列错位
        iy = np.random.randint(0, (h - self.patch_size) // 2) * 2
        ix = np.random.randint(0, (w - self.patch_size) // 2) * 2
        
        patch = img[iy : iy + self.patch_size, ix : ix + self.patch_size]
        
        # 模拟 Bayer RGGB (B:0, G:1, R:2)
        raw = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        raw[0::2, 0::2] = patch[0::2, 0::2, 2] # R
        raw[0::2, 1::2] = patch[0::2, 1::2, 1] # G1
        raw[1::2, 0::2] = patch[1::2, 0::2, 1] # G2
        raw[1::2, 1::2] = patch[1::2, 1::2, 0] # B
        
        # Tensor 转换 (C, H, W)
        input_tensor = torch.from_numpy(raw).float().unsqueeze(0) / 255.0
        target_tensor = torch.from_numpy(patch.transpose(2, 0, 1).copy()).float() / 255.0
        
        return input_tensor, target_tensor

# --- 2. 模型定义 ---
class SimpleISP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1) 
        )
    def forward(self, x):
        return self.net(x)
# --- 2.1 进阶模型定义 ---
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        )

    def forward(self, x):
        return x + self.conv(x)  # 残差连接
class AdvancedISP(nn.Module):
    def __init__(self, num_res_blocks=8):
        super().__init__()
        
        # 1. 特征提取层
        # 将输入从 1 通道提升到 64 通道
        self.head = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 2. 主体残差组
        # 模拟复杂的色彩映射和去马赛克插值
        layers = []
        for _ in range(num_res_blocks):
            layers.append(ResBlock(64))
        self.body = nn.Sequential(*layers)
        
        # 3. 颜色转换层
        # 将 64 维特征映射回 RGB 3 通道
        self.tail = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res  # 全局残差
        x = self.tail(x)
        # 使用 Sigmoid 确保输出在 0-1 之间，防止颜色溢出
        return torch.sigmoid(x)

# --- 3. 训练函数 ---
def train(model, dataloader, optimizer, criterion, epochs=50, val_dataloader=None):
   # 初始化一个无穷大的数作为最初的“最佳分”
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(dataloader)

        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
            val_loss /= len(val_dataloader)
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                #保存这个最牛的模型
                torch.save(model.state_dict(), "best_advanced_isp.pth")
                print(f"🌟 Epoch [{epoch+1}] 表现最佳! 已保存当前权重。")
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
        else:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}")


#测试.pth interface
def test_inference(model, raw_path, device, h=1848, w=2724):
    # 加载已保存的权重
    model_path = os.path.join(os.path.dirname(__file__), "simple_isp_model.pth")
    if not os.path.exists(model_path):
        print("错误：未找到模型文件 simple_isp_model.pth，请先执行训练。")
        return

    #  加载”灵魂”（参数）
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 【重要】切换到评估模式，关闭梯度计算
    
    raw_data = np.fromfile(raw_path, dtype=np.uint8).reshape(h, w)
    
    # 转为 Tensor: [H, W] -> [1, 1, H, W] 并归一化
    input_tensor = torch.from_numpy(raw_data).float().unsqueeze(0).unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)
    
    # 步骤 4: 执行 AI 去马赛克
    with torch.no_grad(): # 推理时不需要记录梯度，节省显存
        output = model(input_tensor)
    
    # 步骤 5: 后处理并保存
    # [1, 3, H, W] -> [H, W, 3]
    result = output.squeeze().cpu().numpy().transpose(1, 2, 0)
    # 模拟人眼感官的 Gamma 曲线
    result = np.power(result, 1.0 / 2.2)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    # 显示和保存
    cv2.imshow("AI ISP Result", cv2.resize(result, (960, 640)))
    cv2.imwrite("ai_final_output.png", result)
    print("AI 推理完成！结果已保存为 ai_final_output.png")
    cv2.waitKey(0)

# --- 4. 主程序入口 ---
if __name__ == "__main__":

    mode ="train"

    torch.backends.cudnn.benchmark = True
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")
    #model = SimpleISP().to(device)
    #进阶版
    model = AdvancedISP(num_res_blocks=8).to(device)

    if mode == "train":
        # 配置
        IMG_DIR = os.path.join(os.path.dirname(__file__), "data/DIV2K_train_LR_bicubic/X2")
        BATCH_SIZE = 32
        EPOCHS = 100
        # 准备数据
        try:
            VAL_DIR = os.path.join(os.path.dirname(__file__), "data/DIV2K_valid_LR_bicubic/X2")
            dataset = DemosaicDataset(IMG_DIR, patch_size=128, patches_per_image=20)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
            val_dataset = DemosaicDataset(VAL_DIR, patch_size=128, patches_per_image=10)
            val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            print("开始训练...")
            train(model, dataloader, optimizer, criterion, epochs=EPOCHS, val_dataloader=val_dataloader)
        
            # 保存模型权重
            torch.save(model.state_dict(), "simple_isp_model.pth")
            print("训练完成，模型已保存为 simple_isp_model.pth")
        
        except Exception as e:
            print(f"运行出错: {e}")
            
    elif mode =="test":
        RAW_PATH = os.path.join(os.path.dirname(__file__), "im0_bgr_2724x1848_rggb_8bit.raw")
        test_inference(model, RAW_PATH, device)
        print(f"完成测试")

