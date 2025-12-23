import numpy as np
import cv2
import os
import heapq
import random
import sys

# --- CẤU HÌNH ---
MAP_SIZE = 256
NUM_SAMPLES = 200
DATA_DIR = "./dataset_dungeon"
SHOW_PREVIEW = True

if not os.path.exists(f"{DATA_DIR}/inputs"): os.makedirs(f"{DATA_DIR}/inputs")
if not os.path.exists(f"{DATA_DIR}/labels"): os.makedirs(f"{DATA_DIR}/labels")

# --- A* SOLVER (Weighted) ---
class Node:
    def __init__(self, x, y, cost=0, h=0, parent=None):
        self.x = x; self.y = y; self.cost = cost; self.h = h; self.parent = parent
    def __lt__(self, other): return (self.cost + self.h) < (other.cost + other.h)

def a_star_search(grid, start, goal):
    rows, cols = grid.shape
    open_list = []
    g_score = {}
    
    # Weight 1.2: Cân bằng giữa tốc độ và đường tối ưu
    start_node = Node(start[0], start[1], 0, np.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2))
    heapq.heappush(open_list, start_node)
    g_score[(start[0], start[1])] = 0
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    iters = 0
    max_iters = 80000 
    
    while open_list:
        iters += 1
        if iters > max_iters: return None

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
                    move_cost = 1.414 if dx!=0 and dy!=0 else 1.0
                    new_cost = current.cost + move_cost
                    if (nx, ny) not in g_score or new_cost < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = new_cost
                        h = np.sqrt((nx-goal[0])**2 + (ny-goal[1])**2)
                        heapq.heappush(open_list, Node(nx, ny, new_cost, h*1.2, current))
    return None

# --- CÁC THUẬT TOÁN ĐÀO HẦM (MINING ALGORITHMS) ---

def create_drunkards_walk(size, num_miners=4, steps=1000):
    """Mô phỏng thợ mỏ say rượu đào hầm (Tạo đường hầm tự nhiên)"""
    grid = np.ones((size, size), dtype=np.uint8) # Toàn bộ là đất đặc (1)
    
    # Các thợ mỏ bắt đầu từ giữa bản đồ
    miners = []
    for _ in range(num_miners):
        miners.append({
            'x': size // 2, 
            'y': size // 2,
            'radius': random.randint(3, 6) # Bán kính đường hầm (Hẹp)
        })
        
    for _ in range(steps):
        for miner in miners:
            # Đào tại vị trí hiện tại
            cv2.circle(grid, (miner['x'], miner['y']), miner['radius'], 0, -1)
            
            # Di chuyển ngẫu nhiên (Random Walk)
            direction = random.randint(0, 3)
            step_len = random.randint(3, 8)
            
            if direction == 0: miner['y'] = max(miner['radius'], miner['y'] - step_len)
            elif direction == 1: miner['y'] = min(size - miner['radius'], miner['y'] + step_len)
            elif direction == 2: miner['x'] = max(miner['radius'], miner['x'] - step_len)
            elif direction == 3: miner['x'] = min(size - miner['radius'], miner['x'] + step_len)
            
            # 5% cơ hội thay đổi kích thước hầm (lúc to lúc nhỏ)
            if random.random() < 0.05:
                miner['radius'] = random.randint(2, 7) # Đôi khi hầm rất hẹp (2px)

    return grid

def create_eroded_ruins(size):
    """Tạo các phòng và hành lang, sau đó làm xói mòn để trông cũ kỹ"""
    # 1. Tạo phòng ngẫu nhiên
    grid = np.ones((size, size), dtype=np.uint8)
    num_rooms = random.randint(10, 20)
    centers = []
    for _ in range(num_rooms):
        w, h = random.randint(15, 40), random.randint(15, 40)
        x, y = random.randint(5, size-w-5), random.randint(5, size-h-5)
        grid[y:y+h, x:x+w] = 0
        centers.append((x+w//2, y+h//2))
    
    # 2. Nối các phòng bằng hành lang zic-zac
    for i in range(len(centers)-1):
        pt1, pt2 = centers[i], centers[i+1]
        cv2.line(grid, pt1, pt2, 0, random.randint(2, 4)) # Hành lang rất hẹp
        
    # 3. Xói mòn (Erosion) để làm nham nhở tường
    # Dùng noise để đục lỗ tường
    noise = (np.random.rand(size, size) < 0.2).astype(np.uint8)
    grid[noise == 1] = 1 # Thêm đất đá vụn vào phòng
    
    # Dùng thuật toán đóng/mở hình thái học để làm mịn cục bộ
    kernel = np.ones((3,3), np.uint8)
    grid = cv2.morphologyEx(grid, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return grid

def add_rubble(grid):
    """Thêm gạch đá vụn rơi vãi (Obstacles nhỏ li ti)"""
    h, w = grid.shape
    # Rải ngẫu nhiên 1000 viên đá nhỏ
    for _ in range(1500):
        rx, ry = random.randint(0, w-1), random.randint(0, h-1)
        if grid[ry, rx] == 0: # Chỉ rải trên đường đi
            grid[ry, rx] = 1 # Đá rơi
    return grid

# --- GENERATOR CHÍNH ---
def generate_dungeon_sample(index):
    # Random chọn kiểu map
    rand_type = random.random()
    
    if rand_type < 0.5:
        # 50% là Mỏ Nhện / Tổ kiến (Random Walk)
        # Nhiều thợ mỏ, đi hướng loạn xạ
        grid_map = create_drunkards_walk(MAP_SIZE, num_miners=random.randint(4, 8), steps=2000)
        map_name = "Spider Mine"
    else:
        # 50% là Di tích đổ nát (Ruins)
        grid_map = create_eroded_ruins(MAP_SIZE)
        map_name = "Eroded Ruins"

    # Thêm độ khó: Rải đá vụn (Rubble)
    grid_map = add_rubble(grid_map)
    
    # Đóng biên
    grid_map[0,:]=1; grid_map[-1,:]=1; grid_map[:,0]=1; grid_map[:,-1]=1

    # Tìm Start/Goal
    # Trong hầm mỏ, start và goal thường ở 2 đầu xa nhất của đường hầm
    path = None
    attempts = 0
    while attempts < 100:
        # Chọn điểm 0 (đường đi) ngẫu nhiên
        free_indices = np.argwhere(grid_map == 0)
        if len(free_indices) < 2: return False # Map lỗi (đặc quá)
        
        start_idx = free_indices[np.random.choice(len(free_indices))]
        goal_idx = free_indices[np.random.choice(len(free_indices))]
        
        start = (start_idx[1], start_idx[0]) # (x, y)
        goal = (goal_idx[1], goal_idx[0])   # (x, y)
        
        dist = np.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2)
        
        if dist > 150: # Phải rất xa nhau
            path = a_star_search(grid_map, start, goal)
            if path: break
        attempts += 1
        
    if path is None: return False

    # Label & Save
    label_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    for (px, py) in path: label_map[py, px] = 1.0
    # Đường đi trong hầm mỏ nên làm label dày một chút (3-5px)
    label_map = cv2.dilate(label_map, np.ones((3,3), np.uint8), iterations=1)
    
    start_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    cv2.circle(start_map, start, 4, 1.0, -1)
    goal_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    cv2.circle(goal_map, goal, 4, 1.0, -1)
    
    input_data = np.stack([grid_map.astype(np.float32), start_map, goal_map], axis=-1)
    np.save(f"{DATA_DIR}/inputs/{index}.npy", input_data)
    np.save(f"{DATA_DIR}/labels/{index}.npy", label_map)

    # Preview
    if SHOW_PREVIEW:
        vis = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        # Màu nâu đất (Dirt color) cho tường
        vis[grid_map == 1] = [40, 70, 100] # BGR: Brownish
        # Màu nền (Đường đi) tối
        vis[grid_map == 0] = [20, 20, 20] 
        
        # Đường đi tìm được màu Vàng đuốc (Torch light)
        vis[label_map > 0] = [0, 200, 255]
        
        cv2.circle(vis, start, 5, (255, 0, 0), -1)
        cv2.circle(vis, goal, 5, (0, 0, 255), -1)
        
        cv2.putText(vis, map_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.imshow("Dungeon Generator", vis)
        cv2.waitKey(50)

    return True

if __name__ == "__main__":
    print(f"Đang đào {NUM_SAMPLES} hầm ngục & mỏ hoang...")
    count = 0
    while count < NUM_SAMPLES:
        if generate_dungeon_sample(count):
            print(f" -> Dungeon {count+1} Mined.")
            count += 1
    cv2.destroyAllWindows()