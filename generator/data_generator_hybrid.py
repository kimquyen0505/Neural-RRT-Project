import numpy as np
import cv2
import os
import heapq
import random
import sys

# --- CẤU HÌNH ---
MAP_SIZE = 256
NUM_SAMPLES = 600
DATA_DIR = "./dataset_hybrid"
SHOW_PREVIEW = True

# Tăng giới hạn đệ quy cho thuật toán maze
sys.setrecursionlimit(5000)

if not os.path.exists(f"{DATA_DIR}/inputs"): os.makedirs(f"{DATA_DIR}/inputs")
if not os.path.exists(f"{DATA_DIR}/labels"): os.makedirs(f"{DATA_DIR}/labels")

# --- A* SOLVER (Weighted x1.5) ---
class Node:
    def __init__(self, x, y, cost=0, h=0, parent=None):
        self.x = x; self.y = y; self.cost = cost; self.h = h; self.parent = parent
    def __lt__(self, other): return (self.cost + self.h) < (other.cost + other.h)

def a_star_search(grid, start, goal):
    rows, cols = grid.shape
    open_list = []
    g_score = {}
    
    start_node = Node(start[0], start[1], 0, np.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2))
    heapq.heappush(open_list, start_node)
    g_score[(start[0], start[1])] = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 4 hướng chính thôi cho nhanh
    
    iters = 0
    while open_list:
        iters += 1
        if iters > 80000: return None

        current = heapq.heappop(open_list)
        if (current.x, current.y) == goal:
            path = []
            while current:
                path.append((current.x, current.y))
                current = current.parent
            return path[::-1]

        if (current.x, current.y) in g_score and g_score[(current.x, current.y)] < current.cost:
            continue

        for dx, dy in directions:
            nx, ny = current.x + dx, current.y + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if grid[ny, nx] == 0:
                    new_cost = current.cost + 1
                    if (nx, ny) not in g_score or new_cost < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = new_cost
                        h = np.sqrt((nx-goal[0])**2 + (ny-goal[1])**2)
                        heapq.heappush(open_list, Node(nx, ny, new_cost, h*1.5, current))
    return None

# --- CÁC "THỢ VẼ" (PAINTERS) ---
# Mỗi hàm này sẽ vẽ lên một vùng (ROI) cụ thể của map

def paint_maze(grid, x, y, w, h):
    """Vẽ mê cung vào vùng chỉ định"""
    # Downscale
    cell_size = 8
    cols = w // cell_size
    rows = h // cell_size
    if cols < 2 or rows < 2: return # Quá nhỏ để vẽ
    
    sub_maze = np.ones((rows, cols), dtype=np.uint8)
    
    def carve(cx, cy):
        sub_maze[cy, cx] = 0
        dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = cx + dx*2, cy + dy*2
            if 0 <= nx < cols and 0 <= ny < rows:
                if sub_maze[ny, nx] == 1:
                    sub_maze[cy + dy, cx + dx] = 0
                    carve(nx, ny)
    try:
        carve(0, 0)
    except: pass # Phòng trường hợp lỗi đệ quy

    # Upscale và chép vào grid chính
    full_sub = cv2.resize(sub_maze, (w, h), interpolation=cv2.INTER_NEAREST)
    grid[y:y+h, x:x+w] = full_sub

def paint_pillars(grid, x, y, w, h):
    """Vẽ rừng cột trụ"""
    spacing = 20
    for iy in range(y + 10, y + h - 10, spacing):
        for ix in range(x + 10, x + w - 10, spacing):
            noise_x = random.randint(-5, 5)
            noise_y = random.randint(-5, 5)
            cx, cy = ix + noise_x, iy + noise_y
            cv2.circle(grid, (cx, cy), random.randint(3, 6), 1, -1)

def paint_rooms(grid, x, y, w, h):
    """Vẽ các phòng vuông vức"""
    # Ban đầu lấp đầy tường
    grid[y:y+h, x:x+w] = 1 
    # Đục phòng
    num_rooms = (w * h) // 1000
    for _ in range(num_rooms):
        rw = random.randint(15, 30)
        rh = random.randint(15, 30)
        rx = random.randint(x, x + w - rw)
        ry = random.randint(y, y + h - rh)
        grid[ry:ry+rh, rx:rx+rw] = 0
        
        # Đục lối đi nối các phòng (Random noise tunnel)
        tunnel_w = random.randint(2, 4)
        if random.random() < 0.5: # Ngang
            cv2.line(grid, (x, ry+rh//2), (x+w, ry+rh//2), 0, tunnel_w)
        else: # Dọc
            cv2.line(grid, (rx+rw//2, y), (rx+rw//2, y+h), 0, tunnel_w)

def paint_clutter(grid, x, y, w, h):
    """Vẽ vật cản lộn xộn (Đá, rác)"""
    num_debris = (w * h) // 100
    for _ in range(num_debris):
        dx = random.randint(x, x+w-1)
        dy = random.randint(y, y+h-1)
        # Vẽ đa giác nhỏ hoặc điểm
        if random.random() < 0.3:
            cv2.circle(grid, (dx, dy), random.randint(2, 5), 1, -1)
        else:
            grid[dy, dx] = 1

# --- CÁC KIỂU LAI TẠO (COMPOSITORS) ---

def create_split_map(size):
    """Chia đôi màn hình: Trái/Phải hoặc Trên/Dưới"""
    grid = np.zeros((size, size), dtype=np.uint8)
    split_pos = size // 2
    is_vertical = random.random() < 0.5
    
    styles = [paint_maze, paint_pillars, paint_rooms, paint_clutter]
    style1, style2 = random.sample(styles, 2)
    
    if is_vertical:
        style1(grid, 0, 0, split_pos, size)          # Trái
        style2(grid, split_pos, 0, size-split_pos, size) # Phải
        # Đục lỗ thông nhau (Gateways)
        for _ in range(3):
            gy = random.randint(20, size-20)
            cv2.rectangle(grid, (split_pos-10, gy), (split_pos+10, gy+15), 0, -1)
    else:
        style1(grid, 0, 0, size, split_pos)          # Trên
        style2(grid, 0, split_pos, size, size-split_pos) # Dưới
        # Đục lỗ
        for _ in range(3):
            gx = random.randint(20, size-20)
            cv2.rectangle(grid, (gx, split_pos-10), (gx+15, split_pos+10), 0, -1)
            
    return grid

def create_quad_map(size):
    """Chia 4 góc 4 kiểu"""
    grid = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    
    styles = [paint_maze, paint_pillars, paint_rooms, paint_clutter]
    random.shuffle(styles)
    
    # Top-Left, Top-Right, Bot-Left, Bot-Right
    styles[0](grid, 0, 0, mid, mid)
    styles[1](grid, mid, 0, size-mid, mid)
    styles[2](grid, 0, mid, mid, size-mid)
    styles[3](grid, mid, mid, size-mid, size-mid)
    
    # Đục lỗ chữ thập ở giữa để thông 4 vùng
    cv2.rectangle(grid, (mid-5, 20), (mid+5, size-20), 0, -1) # Trục dọc
    cv2.rectangle(grid, (20, mid-5), (size-20, mid+5), 0, -1) # Trục ngang
    
    return grid

def create_donut_map(size):
    """Kiểu Bánh Donut: Ngoài là tường dày, Trong là phòng thoáng"""
    grid = np.zeros((size, size), dtype=np.uint8)
    margin = 60 # Độ dày lớp vỏ
    
    # Lớp vỏ (Thường là Maze hoặc Pillars dày)
    outer_style = random.choice([paint_maze, paint_pillars])
    outer_style(grid, 0, 0, size, size)
    
    # Lớp lõi (Xóa rỗng ở giữa rồi vẽ kiểu khác)
    grid[margin:size-margin, margin:size-margin] = 0
    inner_style = random.choice([paint_clutter, paint_rooms])
    inner_style(grid, margin, margin, size-2*margin, size-2*margin)
    
    # Đảm bảo có lối vào lõi (4 cửa)
    mid = size // 2
    cv2.rectangle(grid, (mid-10, 0), (mid+10, margin), 0, -1) # Cửa trên
    cv2.rectangle(grid, (mid-10, size-margin), (mid+10, size), 0, -1) # Cửa dưới
    cv2.rectangle(grid, (0, mid-10), (margin, mid+10), 0, -1) # Cửa trái
    cv2.rectangle(grid, (size-margin, mid-10), (size, mid+10), 0, -1) # Cửa phải
    
    return grid

# --- GENERATOR CHÍNH ---
def generate_hybrid_sample(index):
    rand = random.random()
    if rand < 0.4:
        grid_map = create_split_map(MAP_SIZE)
        m_type = "Split (2 Zones)"
    elif rand < 0.7:
        grid_map = create_quad_map(MAP_SIZE)
        m_type = "Quad (4 Zones)"
    else:
        grid_map = create_donut_map(MAP_SIZE)
        m_type = "Donut (Bunker)"

    # Đóng khung bản đồ
    grid_map[0,:]=1; grid_map[-1,:]=1; grid_map[:,0]=1; grid_map[:,-1]=1

    # Tìm Start/Goal (Cố gắng chọn 2 điểm ở 2 vùng khác nhau)
    path = None
    attempts = 0
    while attempts < 100:
        sx, sy = random.randint(10, MAP_SIZE-10), random.randint(10, MAP_SIZE-10)
        gx, gy = random.randint(10, MAP_SIZE-10), random.randint(10, MAP_SIZE-10)
        
        if grid_map[sy, sx] == 0 and grid_map[gy, gx] == 0:
            dist = np.sqrt((sx-gx)**2 + (sy-gy)**2)
            if dist > 100: # Xa nhau
                path = a_star_search(grid_map, (sx, sy), (gx, gy))
                if path: break
        attempts += 1
        
    if path is None: return False

    # Label & Input
    label_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    for (px, py) in path: label_map[py, px] = 1.0
    label_map = cv2.dilate(label_map, np.ones((3,3), np.uint8), iterations=1)
    
    start_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    cv2.circle(start_map, (sx, sy), 5, 1.0, -1)
    goal_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    cv2.circle(goal_map, (gx, gy), 5, 1.0, -1)
    
    input_data = np.stack([grid_map.astype(np.float32), start_map, goal_map], axis=-1)
    np.save(f"{DATA_DIR}/inputs/{index}.npy", input_data)
    np.save(f"{DATA_DIR}/labels/{index}.npy", label_map)

    # Preview
    if SHOW_PREVIEW:
        vis = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        # Nền màu be nhạt
        vis[:] = [220, 230, 240] 
        # Tường màu tím than
        vis[grid_map == 1] = [60, 40, 60]
        # Đường đi màu xanh ngọc
        vis[label_map > 0] = [200, 200, 0]
        
        cv2.circle(vis, (sx, sy), 6, (0, 0, 255), -1)
        cv2.circle(vis, (gx, gy), 6, (0, 255, 0), -1)
        
        cv2.putText(vis, m_type, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.imshow("Hybrid Map Generator", vis)
        cv2.waitKey(50)

    return True

if __name__ == "__main__":
    print(f"Đang trộn {NUM_SAMPLES} bản đồ Hybrid...")
    count = 0
    while count < NUM_SAMPLES:
        if generate_hybrid_sample(count):
            print(f" -> Hybrid {count+1} Mixed.")
            count += 1
    cv2.destroyAllWindows()