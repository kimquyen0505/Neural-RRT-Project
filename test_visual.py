import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import random
import os

# --- CẤU HÌNH ---
MODEL_PATH = "best_unet_pro.pth"
DATA_DIR = "./Dataset_Ready_For_Train/val"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- ĐỊNH NGHĨA LẠI MODEL (Bắt buộc phải giống hệt lúc train) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.d1 = DoubleConv(in_channels, 64); self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(64, 128); self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(128, 256); self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(256, 512); self.p4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2); self.u1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.u2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.u3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.u4 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        x1 = self.d1(x); p1 = self.p1(x1)
        x2 = self.d2(p1); p2 = self.p2(x2)
        x3 = self.d3(p2); p3 = self.p3(x3)
        x4 = self.d4(p3); p4 = self.p4(x4)
        b = self.bottleneck(p4)
        u1 = self.up1(b); u1 = torch.cat([u1, x4], dim=1); x5 = self.u1(u1)
        u2 = self.up2(x5); u2 = torch.cat([u2, x3], dim=1); x6 = self.u2(u2)
        u3 = self.up3(x6); u3 = torch.cat([u3, x2], dim=1); x7 = self.u3(u3)
        u4 = self.up4(x7); u4 = torch.cat([u4, x1], dim=1); x8 = self.u4(u4)
        return self.out(x8)

def test():
    # Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Lấy danh sách file
    files = glob.glob(f"{DATA_DIR}/inputs/*.npy")
    if not files:
        print("Không tìm thấy data validation!")
        return

    # Random chọn 1 file
    f_path = random.choice(files)
    filename = os.path.basename(f_path)
    l_path = f"{DATA_DIR}/labels/{filename}"
    
    print(f"Testing map: {filename}")
    
    # Load và xử lý input
    inp = np.load(f_path) # (256, 256, 3)
    lbl = np.load(l_path) # (256, 256)
    
    # Resize nếu cần (cho chắc)
    if inp.shape[0] != 256: 
        inp = cv2.resize(inp, (256, 256))
        lbl = cv2.resize(lbl, (256, 256), interpolation=cv2.INTER_NEAREST)
        
    # Chuẩn bị Tensor
    inp_tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    
    # Dự đoán
    with torch.no_grad():
        pred = model(inp_tensor)
        pred = torch.sigmoid(pred) # Đưa về xác suất 0-1
        
    # Chuyển về numpy để vẽ
    pred_img = pred.squeeze().cpu().numpy()
    
    # --- VẼ HÌNH ---
    plt.figure(figsize=(15, 5))
    
    # 1. Input Map (Hiển thị Vật cản + Start + Goal)
    plt.subplot(1, 3, 1)
    # Cộng gộp 3 kênh để hiển thị: Tường (xám) + Start/Goal (sáng)
    vis_map = inp[:,:,0] * 0.5 + inp[:,:,1] + inp[:,:,2]
    plt.imshow(vis_map, cmap='gray')
    plt.title("Input (Map + Start + Goal)")
    plt.axis('off')
    
    # 2. Ground Truth (A*)
    plt.subplot(1, 3, 2)
    plt.imshow(lbl, cmap='jet')
    plt.title("Ground Truth (A*)")
    plt.axis('off')
    
    # 3. AI Prediction
    plt.subplot(1, 3, 3)
    plt.imshow(pred_img, cmap='jet') # Jet map: Xanh (thấp) -> Đỏ (cao)
    plt.title("AI Prediction (U-Net)")
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    test()