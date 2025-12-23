import numpy as np
import cv2
import os
import heapq
import random
import sys

# --- CẤU HÌNH ---
MAP_SIZE = 256
NUM_SAMPLES = 5000 # Số lượng lớn để AI học kỹ
DATA_DIR = "./Dataset_Master_V2" # Tạo folder mới
sys.setrecursionlimit(10000)

if not os.path.exists(f"{DATA_DIR}/inputs"): os.makedirs(f"{DATA_DIR}/inputs")
if not os.path.exists(f"{DATA_DIR}/labels"): os.makedirs(f"{DATA_DIR}/labels")

# --- A* SOLVER ---
class Node:
    def __init__(self, x, y, cost=0, h=0, parent=None):
        self.x, self.y, self.cost, self.h, self.parent = x, y, cost, h, parent
    def __lt__(self, o): return (self.cost + self.h) < (o.cost + o.h)

def a_star(grid, start, goal):
    # A* chuẩn, không heuristic weight để tìm đường ngắn nhất làm mẫu
    open_l, closed_s = [], {}
    heapq.heappush(open_l, Node(start[0], start[1], 0, 0))
    closed_s[(start[0], start[1])] = 0
    dirs = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    iters = 0
    while open_l:
        iters += 1
        if iters > 100000: return None 
        cur = heapq.heappop(open_l)
        if (cur.x, cur.y) == goal:
            path = []
            while cur: path.append((cur.x, cur.y)); cur = cur.parent
            return path[::-1]
        
        for dx, dy in dirs:
            nx, ny = cur.x + dx, cur.y + dy
            if 0<=nx<MAP_SIZE and 0<=ny<MAP_SIZE and grid[ny, nx] == 0:
                cost = 1.414 if dx!=0 and dy!=0 else 1.0
                new_g = cur.cost + cost
                if (nx, ny) not in closed_s or new_g < closed_s[(nx, ny)]:
                    closed_s[(nx, ny)] = new_g
                    h = ((nx-goal[0])**2 + (ny-goal[1])**2)**0.5
                    heapq.heappush(open_l, Node(nx, ny, new_g, h, cur))
    return None

# ==========================================
# CÁC HÀM TẠO MAP (MAP PAINTERS)
# ==========================================

def create_deep_bug_trap(size):
    """BÀI TỦ: Tạo bẫy chữ U sâu để AI học cách đi vòng"""
    grid = np.zeros((size, size), dtype=np.uint8)
    
    # Random vị trí bẫy
    cx, cy = random.randint(50, size-50), random.randint(50, size-50)
    w, h = random.randint(40, 80), random.randint(60, 100) # Bẫy sâu và hẹp
    
    # Vẽ hộp kín
    cv2.rectangle(grid, (cx-w, cy-h), (cx+w, cy+h), 1, -1)
    
    # Khoét rỗng
    thick = random.randint(5, 10)
    cv2.rectangle(grid, (cx-w+thick, cy-h+thick), (cx+w-thick, cy+h-thick), 0, -1)
    
    # Mở cửa (Chỉ 1 hướng duy nhất) -> Ép phải đi vòng
    door_side = random.choice(['top', 'bottom', 'left', 'right'])
    
    # Biến lưu vị trí để đặt Start/Goal hiểm hóc
    trap_center = (cx, cy)
    door_pos = door_side
    
    if door_side == 'top': 
        cv2.rectangle(grid, (cx-w+thick, cy-h), (cx+w-thick, cy-h+thick), 0, -1)
    elif door_side == 'bottom': 
        cv2.rectangle(grid, (cx-w+thick, cy+h-thick), (cx+w-thick, cy+h), 0, -1)
    elif door_side == 'left': 
        cv2.rectangle(grid, (cx-w, cy-h+thick), (cx-w+thick, cy+h-thick), 0, -1)
    elif door_side == 'right': 
        cv2.rectangle(grid, (cx+w-thick, cy-h+thick), (cx+w, cy+h-thick), 0, -1)
        
    return grid, trap_center, door_pos

def create_maze(size):
    """Mê cung đệ quy"""
    cell_size = 16
    h, w = size // cell_size, size // cell_size
    maze = np.ones((h, w), dtype=np.uint8)
    
    def carve(x, y):
        maze[y, x] = 0
        dirs = [(0,1), (0,-1), (1,0), (-1,0)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx*2, y + dy*2
            if 0<=nx<w and 0<=ny<h and maze[ny, nx] == 1:
                maze[y+dy, x+dx] = 0
                carve(nx, ny)
    try: carve(1, 1)
    except: pass
    return cv2.resize(maze, (size, size), interpolation=cv2.INTER_NEAREST), None, None

def create_narrow_passage(size):
    """Khe hẹp tử thần"""
    grid = np.zeros((size, size), dtype=np.uint8)
    
    # Tường ngang hoặc dọc
    if random.random() < 0.5: # Tường ngang
        y = random.randint(50, size-50)
        gap_x = random.randint(20, size-20)
        gap_w = random.randint(4, 10) # Khe siêu hẹp 4-10px
        cv2.rectangle(grid, (0, y), (gap_x, y+10), 1, -1)
        cv2.rectangle(grid, (gap_x+gap_w, y), (size, y+10), 1, -1)
    else: # Tường dọc
        x = random.randint(50, size-50)
        gap_y = random.randint(20, size-20)
        gap_h = random.randint(4, 10)
        cv2.rectangle(grid, (x, 0), (x+10, gap_y), 1, -1)
        cv2.rectangle(grid, (x, gap_y+gap_h), (x+10, size), 1, -1)
        
    return grid, None, None

def create_clutter(size):
    """Vật cản ngẫu nhiên (Dungeon/Urban)"""
    grid = np.zeros((size, size), dtype=np.uint8)
    for _ in range(random.randint(10, 30)):
        if random.random() < 0.5:
            cv2.circle(grid, (random.randint(0, size), random.randint(0, size)), random.randint(10, 30), 1, -1)
        else:
            rx, ry = random.randint(0, size), random.randint(0, size)
            cv2.rectangle(grid, (rx, ry), (rx+random.randint(20, 50), ry+random.randint(20, 50)), 1, -1)
    return grid, None, None

# ==========================================
# MAIN LOOP
# ==========================================
def generate_master():
    print(f"🚀 Bắt đầu tạo {NUM_SAMPLES} map tổng hợp...")
    
    for i in range(NUM_SAMPLES):
        # Chọn loại map (Ưu tiên Bug Trap và Narrow để AI học kỹ)
        rand = random.random()
        trap_info = None
        
        if rand < 0.35: # 35% là Deep Bug Trap (Học bài tủ)
            grid, trap_center, door_pos = create_deep_bug_trap(MAP_SIZE)
            trap_info = (trap_center, door_pos)
            m_type = "Trap"
        elif rand < 0.55: # 20% là Narrow Passage
            grid, _, _ = create_narrow_passage(MAP_SIZE)
            m_type = "Narrow"
        elif rand < 0.75: # 20% là Maze
            grid, _, _ = create_maze(MAP_SIZE)
            m_type = "Maze"
        else: # 25% là Clutter/Urban
            grid, _, _ = create_clutter(MAP_SIZE)
            m_type = "Clutter"

        # Chọn Start/Goal thông minh
        path = None
        attempts = 0
        while attempts < 20:
            sx, sy = random.randint(10, MAP_SIZE-10), random.randint(10, MAP_SIZE-10)
            gx, gy = random.randint(10, MAP_SIZE-10), random.randint(10, MAP_SIZE-10)
            
            # Nếu là Trap Map: Cố tình đặt Start trong bẫy, Goal ngoài bẫy (đối diện cửa)
            if m_type == "Trap" and trap_info and attempts < 10:
                center, door = trap_info
                # Start gần tâm bẫy
                sx = np.clip(center[0] + random.randint(-10, 10), 0, MAP_SIZE-1)
                sy = np.clip(center[1] + random.randint(-10, 10), 0, MAP_SIZE-1)
                
                # Goal ở phía đối diện cửa ra -> Ép đi vòng
                if door == 'top': gy = min(center[1] + 100, MAP_SIZE-10)
                elif door == 'bottom': gy = max(center[1] - 100, 10)
                elif door == 'left': gx = min(center[0] + 100, MAP_SIZE-10)
                elif door == 'right': gx = max(center[0] - 100, 10)

            # Check hợp lệ
            if grid[sy, sx] == 0 and grid[gy, gx] == 0:
                dist = ((sx-gx)**2 + (sy-gy)**2)**0.5
                if dist > 50: # Khoảng cách đủ xa
                    path = a_star(grid, (sx, sy), (gx, gy))
                    if path: break
            attempts += 1
        
        if path is None: continue 

        # Lưu dữ liệu
        lbl = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
        for p in path: lbl[p[1], p[0]] = 1.0
        lbl = cv2.dilate(lbl, np.ones((5,5), np.uint8), iterations=1) # Label đậm 5px

        s_map = np.zeros_like(grid, dtype=np.float32); cv2.circle(s_map, (sx,sy), 8, 1, -1)
        g_map = np.zeros_like(grid, dtype=np.float32); cv2.circle(g_map, (gx,gy), 8, 1, -1)
        inp = np.stack([grid.astype(np.float32), s_map, g_map], axis=-1)

        np.save(f"{DATA_DIR}/inputs/{i}.npy", inp)
        np.save(f"{DATA_DIR}/labels/{i}.npy", lbl)
        
        if i % 100 == 0: print(f" -> Generated {i}/{NUM_SAMPLES} ({m_type})")

    print("✅ Hoàn tất Master Dataset V2!")

if __name__ == "__main__":
    generate_master()