import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
import random
import math
import time
import os

# ==========================================
# 1. KIẾN TRÚC MẠNG U-NET (Từ Thành viên 2)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_c=3, out_c=1):
        super().__init__()
        self.d1 = DoubleConv(in_c, 64); self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(64, 128);  self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(128, 256); self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(256, 512); self.p4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2); self.u1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2);  self.u2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2);  self.u3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2);   self.u4 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, out_c, 1)
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

# ==========================================
# 2. THUẬT TOÁN NEURAL RRT* (BẢN FULL)
# ==========================================
class Node:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.parent = None; self.cost = 0.0

class NeuralRRTStar:
    def __init__(self, occupancy_grid, start, goal, model=None, device="cpu", max_iter=150000, step_size=3):
        self.grid = occupancy_grid
        self.height, self.width = occupancy_grid.shape
        self.device = device
        self.start = Node(*self.clamp_to_free(start[0], start[1]))
        self.goal = Node(*self.clamp_to_free(goal[0], goal[1]))
        self.max_iter = max_iter
        self.step_size = step_size
        self.node_list = [self.start]
        self.model = model
        self.prob_map = None
        if self.model: self.prob_map = self.generate_prob_map()

    def clamp_to_free(self, x, y):
        for r in range(10):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    nx, ny = int(x+dx), int(y+dy)
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if self.grid[ny, nx] < 0.5: return nx, ny
        return x, y

    def generate_prob_map(self):
        s_map = np.zeros_like(self.grid, dtype=np.float32); cv2.circle(s_map, (self.start.x, self.start.y), 8, 1.0, -1)
        g_map = np.zeros_like(self.grid, dtype=np.float32); cv2.circle(g_map, (self.goal.x, self.goal.y), 8, 1.0, -1)
        inp = np.stack([self.grid.astype(np.float32), s_map, g_map], axis=-1)
        inp_small = cv2.resize(inp, (128, 128))
        inp_tensor = torch.from_numpy(inp_small).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            pred = self.model(inp_tensor)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
        return cv2.resize(pred, (self.width, self.height))

    def get_sample(self, current_iter):
        p = random.random()
        # Chiến thuật Exploration tăng dần theo thời gian
        ai_ratio = max(0.1, 0.4 - (current_iter / self.max_iter) * 0.3)
        if self.prob_map is not None and p < ai_ratio: 
            for _ in range(10):
                rx, ry = random.randint(0, self.width-1), random.randint(0, self.height-1)
                if self.prob_map[ry, rx] > 0.05: return rx, ry
        if p < ai_ratio + 0.05: return self.goal.x, self.goal.y
        return random.randint(0, self.width-1), random.randint(0, self.height-1)

    def is_collision(self, n1, n2):
        dist = math.hypot(n2.x - n1.x, n2.y - n1.y)
        steps = int(dist) + 1
        line_pts = np.linspace([n1.x, n1.y], [n2.x, n2.y], num=steps)
        for pt in line_pts:
            ix, iy = int(pt[0]), int(pt[1])
            if 0 <= ix < self.width and 0 <= iy < self.height:
                if self.grid[iy, ix] > 0.5: return True
            else: return True
        return False

    def plan(self):
        print(f"--- Đang giải mê cung (Max: {self.max_iter}) ---")
        for i in range(self.max_iter):
            rx, ry = self.get_sample(i)
            nearest = min(self.node_list, key=lambda n: (n.x - rx)**2 + (n.y - ry)**2)
            theta = math.atan2(ry - nearest.y, rx - nearest.x)
            new_node = Node(int(nearest.x + self.step_size * math.cos(theta)),
                            int(nearest.y + self.step_size * math.sin(theta)))

            if 0 <= new_node.x < self.width and 0 <= new_node.y < self.height:
                if not self.is_collision(nearest, new_node):
                    new_node.parent = nearest
                    new_node.cost = nearest.cost + self.step_size
                    self.node_list.append(new_node)
                    if math.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) < self.step_size * 2:
                        self.goal.parent = new_node
                        return self.extract_path(), i # <-- LỖI Ở ĐÂY ĐÃ ĐƯỢC FIX
            if i % 10000 == 0 and i > 0: print(f"Đã thử {i} vòng...")
        return None, self.max_iter

    # --- ĐÃ THÊM LẠI CÁC HÀM BỊ THIẾU ---
    def extract_path(self):
        path = []; n = self.goal
        while n:
            path.append([n.x, n.y]); n = n.parent
        return path[::-1]

    def smooth_path(self, path):
        if not path or len(path) < 3: return path
        smoothed = [path[0]]; curr = 0
        while curr < len(path) - 1:
            best_next = curr + 1
            for next_idx in range(len(path)-1, curr + 1, -1):
                if not self.is_collision(Node(*path[curr]), Node(*path[next_idx])):
                    best_next = next_idx; break
            smoothed.append(path[best_next]); curr = best_next
        return smoothed

# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    INPUT_FILE = "dataset_all/dataset_cau_truc/inputs/98.npy" 
    MODEL_PATH = "AI_model.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("✅ Load Model OK!")
    else: print("❌ Thiếu AI_model.pth"); exit()

    data = np.load(INPUT_FILE)
    obs_map = data[:,:,0] if data.ndim == 3 else data
    label_p = INPUT_FILE.replace("inputs", "labels")
    label_data = np.load(label_p)
    ly, lx = np.where(label_data > 0); start_p, goal_p = (lx[0], ly[0]), (lx[-1], ly[-1])

    planner = NeuralRRTStar(obs_map, start_p, goal_p, model=model, device=DEVICE, max_iter=150000, step_size=3)
    st = time.time()
    path, iterations = planner.plan()
    et = time.time()

    vis = cv2.cvtColor((obs_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    vis = cv2.bitwise_not(vis)
    if planner.prob_map is not None:
        heatmap = (planner.prob_map * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        vis = cv2.addWeighted(vis, 0.7, heatmap_color, 0.3, 0)

    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.scatter(start_p[0], start_p[1], c='g', s=100); plt.scatter(goal_p[0], goal_p[1], c='b', s=100)

    if path:
        smoothed = planner.smooth_path(path)
        path, smoothed = np.array(path), np.array(smoothed)
        plt.plot(path[:,0], path[:,1], 'w--', alpha=0.3)
        plt.plot(smoothed[:,0], smoothed[:,1], 'r-', linewidth=3, label='Neural Path')
        print(f"THÀNH CÔNG! Time: {et-st:.2f}s | Iterations: {iterations}")
        plt.title(f"Success! {et-st:.2f}s | Iters: {iterations}")
    else:
        print("Vẫn thất bại."); plt.title("Failed")
    plt.show()