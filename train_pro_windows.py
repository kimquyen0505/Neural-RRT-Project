import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# Cập nhật import mới cho PyTorch 2.x+
from torch.amp import autocast, GradScaler 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import os
import cv2

# --- CẤU HÌNH "KHỦNG" CHO RTX 5070 Ti ---
BATCH_SIZE = 64          
LEARNING_RATE = 1e-4     
EPOCHS = 100             
DATA_DIR = "./Dataset_Ready_For_Train" 
SAVE_PATH = "best_unet_pro.pth"

# Kiểm tra GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. DATASET LOADER (Tối ưu I/O) ---
class PathPlanningDataset(Dataset):
    def __init__(self, root_dir):
        self.input_paths = sorted(glob.glob(f"{root_dir}/inputs/*.npy"))
        self.label_paths = sorted(glob.glob(f"{root_dir}/labels/*.npy"))
        
    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, idx):
        try:
            # Load npy
            inp = np.load(self.input_paths[idx]) 
            lbl = np.load(self.label_paths[idx])
            
            # Resize an toàn
            if inp.shape[0] != 256 or inp.shape[1] != 256:
                inp = cv2.resize(inp, (256, 256))
                lbl = cv2.resize(lbl, (256, 256), interpolation=cv2.INTER_NEAREST)

            # Fix lỗi Windows Multi-processing bằng .copy()
            inp = inp.copy()
            lbl = lbl.copy()
            
            # Chuyển Tensor
            inp_tensor = torch.from_numpy(inp).permute(2, 0, 1).float()
            lbl_tensor = torch.from_numpy(lbl).unsqueeze(0).float()
            
            return inp_tensor, lbl_tensor
        except Exception as e:
            # Tránh crash nếu 1 file lỗi
            print(f"Lỗi file {idx}: {e}")
            return torch.zeros(3, 256, 256), torch.zeros(1, 256, 256)

# --- 2. MODEL U-NET ---
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

# --- 3. TRAINING LOOP (PRO VERSION - FIXED) ---
def train():
    print(f"🚀 KHỞI ĐỘNG SUPER TRAINING TRÊN: {DEVICE}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Dataset
    train_ds = PathPlanningDataset(f"{DATA_DIR}/train")
    val_ds = PathPlanningDataset(f"{DATA_DIR}/val")
    
    # DataLoader tối ưu cho i7 14700K (8 workers)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=8, pin_memory=True, persistent_workers=True)
    
    print(f"📊 Dữ liệu: {len(train_ds)} Train | {len(val_ds)} Val")
    
    model = UNet().to(DEVICE)
    
    # Mixed Precision Scaler (Cú pháp mới cho torch.amp)
    scaler = GradScaler('cuda') 
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- FIX LỖI Ở ĐÂY: Bỏ verbose=True ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # --- TĂNG TỐC VỚI AMP (Automatic Mixed Precision) ---
            with autocast('cuda'): 
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Cập nhật Scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr'] # Tự in Learning Rate ra để theo dõi
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f" -> Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | LR: {current_lr}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print("💾 New Best Model Saved!")

    print("✅ TRAINING COMPLETED!")
    
    # Lưu biểu đồ
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Training with RTX 5070 Ti - Best Loss: {best_val_loss:.4f}')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.savefig("training_result.png")
    plt.show()

# --- ENTRY POINT CHO WINDOWS ---
if __name__ == '__main__':
    # Fix lỗi thư viện xung đột (nếu có)
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    train()