import numpy as np
import cv2
import os
import heapq
import random
import math
import sys

# Tăng giới hạn đệ quy cho Fractal và Bio-tree
sys.setrecursionlimit(10000)

# --- CẤU HÌNH ---
MAP_SIZE = 256
NUM_SAMPLES = 500
DATA_DIR = "./dataset_ultra"
SHOW_PREVIEW = True

if not os.path.exists(f"{DATA_DIR}/inputs"): os.makedirs(f"{DATA_DIR}/inputs")
if not os.path.exists(f"{DATA_DIR}/labels"): os.makedirs(f"{DATA_DIR}/labels")

# --- A* SOLVER (Weighted x1.3) ---
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
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    iters = 0
    while open_list:
        iters += 1
        if iters > 100000: return None

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
                        heapq.heappush(open_list, Node(nx, ny, new_cost, h*1.3, current))
    return None

# --- CÁC THUẬT TOÁN ĐỊA HÌNH ULTRA ---

def create_fractal_ruins(size):
    """Tạo map Fractal (Sierpinski Carpet)"""
    grid = np.zeros((size, size), dtype=np.uint8)
    
    def recursive_squares(x, y, s):
        if s < 10: return # Dừng khi ô quá nhỏ
        
        # Chia thành lưới 3x3
        sub_s = s // 3
        
        # Duyệt qua 9 ô con
        for i in range(3):
            for j in range(3):
                # Ô ở giữa (1,1) sẽ là vật cản
                if i == 1 and j == 1:
                    # Random: Đôi khi không vẽ để tạo lỗ hổng ngẫu nhiên
                    if random.random() < 0.9: 
                        cv2.rectangle(grid, (x + sub_s, y + sub_s), 
                                      (x + 2*sub_s, y + 2*sub_s), 1, -1)
                else:
                    # Các ô xung quanh tiếp tục đệ quy
                    recursive_squares(x + i*sub_s, y + j*sub_s, sub_s)

    # Vẽ khung ngoài
    recursive_squares(0, 0, size)
    
    # Đục thêm lỗ lớn ngẫu nhiên để đảm bảo tính liên thông
    for _ in range(5):
        rx, ry = random.randint(0, size), random.randint(0, size)
        cv2.circle(grid, (rx, ry), 20, 0, -1)
        
    return grid

def create_bio_network(size):
    """Tạo mạng lưới sinh học (Rễ cây / Mạch máu)"""
    # Ban đầu là đặc (1)
    grid = np.ones((size, size), dtype=np.uint8)
    
    def grow_branch(x, y, angle, width, depth):
        if width < 1 or depth <= 0: return
        if not (0 <= x < size and 0 <= y < size): return
        
        # Vẽ đoạn thẳng (Đường hầm)
        length = random.randint(10, 20)
        end_x = int(x + length * math.cos(angle))
        end_y = int(y + length * math.sin(angle))
        
        cv2.line(grid, (int(x), int(y)), (end_x, end_y), 0, int(width))
        
        # Đệ quy chia nhánh
        num_branches = random.randint(1, 3)
        for _ in range(num_branches):
            new_angle = angle + random.uniform(-0.8, 0.8) # Góc lệch ngẫu nhiên
            new_width = width * 0.8 # Nhánh nhỏ dần
            grow_branch(end_x, end_y, new_angle, new_width, depth - 1)

    # Bắt đầu từ tâm và 4 góc
    centers = [(size//2, size//2), (0,0), (size,size), (0,size), (size,0)]
    for cx, cy in centers:
        for _ in range(3): # Mỗi tâm mọc ra vài nhánh cái
            start_angle = random.uniform(0, 2*math.pi)
            grow_branch(cx, cy, start_angle, width=15, depth=8)
            
    return grid

def create_honeycomb(size):
    """Tạo lưới tổ ong (Hexagonal)"""
    grid = np.zeros((size, size), dtype=np.uint8)
    
    radius = 15
    # Công thức tọa độ lục giác
    dx = radius * 3 / 2
    dy = radius * math.sqrt(3)
    
    rows = int(size / dy) + 2
    cols = int(size / dx) + 2
    
    for r in range(rows):
        for c in range(cols):
            # Tính tâm lục giác
            x = c * dx
            y = r * dy
            if c % 2 == 1: y += dy / 2
            
            x, y = int(x), int(y)
            
            # 40% ô là vật cản
            if random.random() < 0.4:
                # Vẽ lục giác
                pts = []
                for i in range(6):
                    angle_deg = 60 * i
                    angle_rad = math.pi / 180 * angle_deg
                    px = int(x + radius * math.cos(angle_rad))
                    py = int(y + radius * math.sin(angle_rad))
                    pts.append([px, py])
                
                pts = np.array(pts, np.int32)
                cv2.fillPoly(grid, [pts], 1)
                
    return grid

def create_mirror_chamber(size):
    """Tạo phòng gương (Đối xứng 4 góc)"""
    grid = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    
    # Chỉ vẽ ở góc phần tư thứ nhất (Top-Left)
    roi_size = mid
    
    # Vẽ các hình ngẫu nhiên vào góc này
    for _ in range(random.randint(5, 10)):
        shape_t = random.choice(['rect', 'circle', 'tri'])
        x = random.randint(0, roi_size)
        y = random.randint(0, roi_size)
        s = random.randint(10, 40)
        
        if shape_t == 'rect':
            cv2.rectangle(grid, (x, y), (x+s, y+s), 1, -1)
        elif shape_t == 'circle':
            cv2.circle(grid, (x, y), s//2, 1, -1)
        elif shape_t == 'tri':
            pts = np.array([[x, y], [x+s, y], [x+s//2, y-s]], np.int32)
            cv2.fillPoly(grid, [pts], 1)
            
    # Lấy góc phần tư đã vẽ
    top_left = grid[0:mid, 0:mid]
    
    # Đối xứng ngang (Mirror X)
    top_right = cv2.flip(top_left, 1)
    grid[0:mid, mid:size] = top_right
    
    # Đối xứng dọc (Mirror Y) cho cả nửa trên
    top_half = grid[0:mid, 0:size]
    bot_half = cv2.flip(top_half, 0)
    grid[mid:size, 0:size] = bot_half
    
    # Vẽ tường bao quanh
    grid[0:5,:]=1; grid[-5:,:]=1; grid[:,0:5]=1; grid[:,-5:]=1
    
    return grid

# --- GENERATOR CHÍNH ---
def generate_ultra_sample(index):
    rand = random.random()
    if rand < 0.25:
        grid_map = create_fractal_ruins(MAP_SIZE)
        m_type = "Fractal (De quy)"
    elif rand < 0.5:
        grid_map = create_bio_network(MAP_SIZE)
        m_type = "Bio-Network (Mach mau)"
    elif rand < 0.75:
        grid_map = create_honeycomb(MAP_SIZE)
        m_type = "Honeycomb (Luc giac)"
    else:
        grid_map = create_mirror_chamber(MAP_SIZE)
        m_type = "Mirror (Doi xung)"

    # Tìm Start/Goal
    path = None
    attempts = 0
    while attempts < 200:
        sx, sy = random.randint(10, MAP_SIZE-10), random.randint(10, MAP_SIZE-10)
        gx, gy = random.randint(10, MAP_SIZE-10), random.randint(10, MAP_SIZE-10)
        
        if grid_map[sy, sx] == 0 and grid_map[gy, gx] == 0:
            dist = np.sqrt((sx-gx)**2 + (sy-gy)**2)
            if dist > 80:
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
        
        if "Fractal" in m_type:
            vis[:] = [20, 0, 0] # Nền đỏ thẫm
            vis[grid_map==1] = [0, 255, 255] # Fractal Vàng
        elif "Bio" in m_type:
            vis[:] = [100, 50, 50] # Nền thịt/đất
            vis[grid_map==0] = [20, 20, 20]  # Mạch máu đen
        elif "Honeycomb" in m_type:
            vis[:] = [255, 200, 0] # Vàng mật ong
            vis[grid_map==1] = [50, 30, 0] # Sáp nâu
        else: # Mirror
            vis[:] = [255, 255, 255]
            vis[grid_map==1] = [0, 0, 0]
            
        vis[label_map > 0] = [0, 255, 0] if "Bio" not in m_type else [0, 255, 255]
        
        cv2.circle(vis, (sx, sy), 5, (255, 255, 255), -1)
        cv2.circle(vis, (gx, gy), 5, (255, 0, 0), -1)
        
        cv2.putText(vis, m_type, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,255,100), 1)
        cv2.imshow("Ultra Generator", vis)
        cv2.waitKey(50)

    return True

if __name__ == "__main__":
    print(f"Đang tạo {NUM_SAMPLES} bản đồ Siêu Phức Tạp...")
    count = 0
    while count < NUM_SAMPLES:
        if generate_ultra_sample(count):
            print(f" -> Ultra {count+1} Generated.")
            count += 1
    cv2.destroyAllWindows()