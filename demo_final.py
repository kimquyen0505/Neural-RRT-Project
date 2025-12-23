import torch
import torch.nn as nn
import numpy as np
import cv2
import random
import math
import time
import glob
import heapq 
import scipy.interpolate as interpolate
import os # <--- Thư viện để quản lý file/folder

# --- CẤU HÌNH ---
MAP_SIZE = 256
MODEL_INPUT_SIZE = 128 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULT_DIR = "./demo_results" # Tên folder lưu kết quả

# ==========================================
# 1. BỘ KIỂM TRA ĐƯỜNG ĐI
# ==========================================
def check_connectivity(grid, start, goal):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    gscore = {start:0}
    fscore = {start:math.hypot(start[0]-goal[0], start[1]-goal[1])}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal: return True
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            if 0 <= neighbor[0] < grid.shape[1] and 0 <= neighbor[1] < grid.shape[0]:
                if grid[neighbor[1]][neighbor[0]] == 1: continue
            else: continue
            if neighbor in close_set: continue
            
            tentative_g = gscore[current] + math.hypot(i, j)
            if neighbor not in gscore or tentative_g < gscore[neighbor]:
                gscore[neighbor] = tentative_g
                fscore[neighbor] = tentative_g + math.hypot(neighbor[0]-goal[0], neighbor[1]-goal[1])
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return False

# ==========================================
# 2. MODEL U-NET (Standard)
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
        self.d1 = DoubleConv(in_c, 64);  self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(64, 128);   self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(128, 256);  self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(256, 512);  self.p4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2); self.u1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.u2 = DoubleConv(512, 256)
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
# 3. CÁC HÀM HỖ TRỢ
# ==========================================
def create_organic_cavern(size):
    grid = (np.random.rand(size, size) > 0.55).astype(np.uint8) 
    for _ in range(5):
        new_grid = grid.copy()
        for y in range(1, size-1):
            for x in range(1, size-1):
                neighbors = np.sum(grid[y-1:y+2, x-1:x+2]) - grid[y, x]
                if neighbors > 4: new_grid[y, x] = 1
                elif neighbors < 4: new_grid[y, x] = 0
        grid = new_grid
    grid[0:5,:] = 1; grid[-5:,:] = 1; grid[:,0:5] = 1; grid[:,-5:] = 1
    return grid

def smooth_path(path):
    if len(path) < 3: return path
    filtered = [path[0]]
    for i in range(1, len(path)):
        if math.hypot(path[i][0]-filtered[-1][0], path[i][1]-filtered[-1][1]) > 5:
            filtered.append(path[i])
    if len(filtered) < 3: return path
    try:
        x = [p[0] for p in filtered]; y = [p[1] for p in filtered]
        tck, u = interpolate.splprep([x, y], s=3, k=2)
        u_new = np.linspace(0, 1, num=len(filtered)*5)
        x_new, y_new = interpolate.splev(u_new, tck)
        return list(zip([int(i) for i in x_new], [int(i) for i in y_new]))
    except: return path

class Node:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.parent = None; self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, obstacle_map, model=None, use_ai=False):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.map = obstacle_map
        self.nodes = [self.start]
        self.model = model
        self.use_ai = use_ai
        self.prob_map = None
        if self.use_ai and self.model:
            self.prob_map = self.generate_prob_map(start, goal)

    def generate_prob_map(self, start, goal):
        s_map = np.zeros_like(self.map, dtype=np.float32); cv2.circle(s_map, start, 8, 1.0, -1)
        g_map = np.zeros_like(self.map, dtype=np.float32); cv2.circle(g_map, goal, 8, 1.0, -1)
        inp = np.stack([self.map.astype(np.float32), s_map, g_map], axis=-1)
        inp_small = cv2.resize(inp, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        inp_tensor = torch.from_numpy(inp_small).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        
        with torch.no_grad():
            pred = self.model(inp_tensor)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
            
        return cv2.resize(pred, (MAP_SIZE, MAP_SIZE))

    def get_sample(self):
        if self.use_ai and self.prob_map is not None:
            if random.random() < 0.60: 
                for _ in range(20):
                    rx = random.randint(0, MAP_SIZE-1); ry = random.randint(0, MAP_SIZE-1)
                    if self.prob_map[ry, rx] > 0.05 and random.random() < self.prob_map[ry, rx]:
                        return rx, ry
        return random.randint(0, MAP_SIZE-1), random.randint(0, MAP_SIZE-1)

    def planning(self, max_iter=10000):
        step_size = 10; search_radius = 25
        for i in range(max_iter):
            rnd = self.get_sample()
            nearest = min(self.nodes, key=lambda n: math.hypot(n.x - rnd[0], n.y - rnd[1]))
            theta = math.atan2(rnd[1] - nearest.y, rnd[0] - nearest.x)
            new_x = int(nearest.x + step_size * math.cos(theta))
            new_y = int(nearest.y + step_size * math.sin(theta))
            
            if not (0 <= new_x < MAP_SIZE and 0 <= new_y < MAP_SIZE): continue
            if self.map[new_y, new_x] == 1: continue 
            if not self.check_collision(nearest, new_x, new_y): continue
            
            new_node = Node(new_x, new_y)
            new_node.parent = nearest
            new_node.cost = nearest.cost + step_size
            self.nodes.append(new_node)
            
            neighbors = [n for n in self.nodes if math.hypot(n.x - new_x, n.y - new_y) < search_radius]
            for nb in neighbors:
                if new_node.cost + math.hypot(nb.x - new_x, nb.y - new_y) < nb.cost:
                    if self.check_collision(new_node, nb.x, nb.y):
                        nb.parent = new_node
                        nb.cost = new_node.cost + math.hypot(nb.x - new_x, nb.y - new_y)

            if math.hypot(new_x - self.goal.x, new_y - self.goal.y) < 15:
                print(f"[{'AI' if self.use_ai else 'STD'}] Found path at iter {i}")
                self.goal.parent = new_node
                return self.extract_path()
        return None

    def check_collision(self, node, x2, y2):
        dist = math.hypot(x2 - node.x, y2 - node.y)
        steps = int(dist / 2) + 1
        for i in range(steps):
            t = i / steps
            cx = int(node.x + (x2 - node.x) * t); cy = int(node.y + (y2 - node.y) * t)
            if self.map[cy, cx] == 1: return False
        return True

    def extract_path(self):
        path = []; node = self.goal
        while node.parent:
            path.append((node.x, node.y)); node = node.parent
        path.append((self.start.x, self.start.y))
        return path[::-1]

# ==========================================
# 5. MAIN
# ==========================================
def main():
    # --- TẠO FOLDER LƯU KẾT QUẢ ---
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        print(f"📁 Đã tạo folder lưu kết quả: {RESULT_DIR}")

    pth_files = glob.glob("*.pth")
    if not pth_files: print("❌ Không tìm thấy model"); return
    MODEL_PATH = pth_files[0]
    
    print(f"👉 Dùng model: {MODEL_PATH}")
    try:
        model = UNet().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print("✅ Model OK!")
    except Exception as e: print(f"❌ Lỗi: {e}"); return

    # --- VÒNG LẶP SINH MAP ---
    print("🛠 Đang sinh map HANG ĐỘNG (và kiểm tra đường đi)...")
    
    attempts_map = 0
    while True:
        attempts_map += 1
        grid = create_organic_cavern(MAP_SIZE)
        s_cand, g_cand = None, None
        
        for _ in range(100):
            s = (random.randint(20, 230), random.randint(20, 230))
            g = (random.randint(20, 230), random.randint(20, 230))
            if grid[s[1], s[0]] == 0 and grid[g[1], g[0]] == 0:
                if math.hypot(s[0]-g[0], s[1]-g[1]) > 150:
                    if check_connectivity(grid, s, g):
                        s_cand, g_cand = s, g
                        break 
        
        if s_cand and g_cand:
            start, goal = s_cand, g_cand
            print(f"✅ Đã tìm thấy map hợp lệ sau {attempts_map} lần thử!")
            break
        else:
            print(f"⚠️ Map {attempts_map} là đường cụt...", end="\r")

    cv2.circle(grid, start, 5, 0, -1); cv2.circle(grid, goal, 5, 0, -1)

    print("\n--- Chạy RRT* Thường ---")
    t0 = time.time()
    rrt_std = RRTStar(start, goal, grid, use_ai=False)
    path_std = rrt_std.planning(max_iter=30000) 
    t_std = time.time() - t0
    
    print("\n--- Chạy Neural RRT* (AI) ---")
    t0 = time.time()
    rrt_ai = RRTStar(start, goal, grid, model=model, use_ai=True)
    path_ai = rrt_ai.planning(max_iter=15000) 
    t_ai = time.time() - t0

    # --- VẼ ---
    vis = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    vis[grid == 0] = [30, 30, 30]; vis[grid == 1] = [50, 100, 160] 
    
    if rrt_ai.prob_map is not None:
        hm = (rrt_ai.prob_map * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        vis = cv2.addWeighted(vis, 0.6, hm_color, 0.4, 0)

    for n in rrt_std.nodes:
        if n.parent: cv2.line(vis, (int(n.x), int(n.y)), (int(n.parent.x), int(n.parent.y)), (150, 150, 150), 1)
    for n in rrt_ai.nodes:
        if n.parent: cv2.line(vis, (int(n.x), int(n.y)), (int(n.parent.x), int(n.parent.y)), (0, 255, 255), 1)

    if path_std:
        for i in range(len(path_std)-1): cv2.line(vis, path_std[i], path_std[i+1], (0, 0, 255), 2)
    if path_ai:
        sm_ai = smooth_path(path_ai)
        for i in range(len(sm_ai)-1): cv2.line(vis, sm_ai[i], sm_ai[i+1], (50, 255, 50), 3)

    cv2.circle(vis, start, 8, (255, 0, 255), -1); cv2.circle(vis, goal, 8, (0, 255, 255), -1)
    
    print(f"\nKẾT QUẢ CUỐI CÙNG:")
    print(f"Standard RRT*: {t_std:.4f}s")
    print(f"Neural RRT*  : {t_ai:.4f}s")
    
    cv2.putText(vis, f"Std Time: {t_std:.3f}s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.putText(vis, f"AI Time: {t_ai:.3f}s", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,255,50), 2)
    
    # --- LƯU ẢNH ---
    # Tạo tên file theo thời gian thực (năm-tháng-ngày_giờ-phút-giây)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{RESULT_DIR}/result_{timestamp}.png"
    cv2.imwrite(filename, vis)
    print(f"💾 Đã lưu kết quả vào: {filename}")

    cv2.imshow("Smart Demo (Saved)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()