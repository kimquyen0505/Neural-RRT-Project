import numpy as np
import cv2
import os
import heapq
import random
import math

# --- CẤU HÌNH ---
MAP_SIZE = 256
NUM_SAMPLES = 500
DATA_DIR = "./dataset_hinh_hoc"
SHOW_PREVIEW = True

if not os.path.exists(f"{DATA_DIR}/inputs"): os.makedirs(f"{DATA_DIR}/inputs")
if not os.path.exists(f"{DATA_DIR}/labels"): os.makedirs(f"{DATA_DIR}/labels")

# --- A* SOLVER (Weighted x1.5 cho nhanh) ---
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
    # Map này rất khó nên cho phép tìm kiếm lâu hơn
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
                        heapq.heappush(open_list, Node(nx, ny, new_cost, h*1.5, current))
    return None

# --- CÁC THUẬT TOÁN CHAOS ---

def create_spiral(size):
    """Tạo mê cung xoắn ốc"""
    grid = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    
    # Vẽ đường xoắn ốc
    # Công thức: x = a + b*angle * cos(angle)
    num_turns = random.randint(3, 6)
    max_angle = num_turns * 2 * math.pi
    thickness = random.randint(3, 6)
    spacing = random.randint(15, 25) # Khoảng cách giữa các vòng
    
    prev_x, prev_y = cx, cy
    
    for angle in np.arange(0, max_angle, 0.05):
        r = spacing * angle / (2*math.pi)
        x = int(cx + r * math.cos(angle))
        y = int(cy + r * math.sin(angle))
        
        if 0 <= x < size and 0 <= y < size:
            cv2.line(grid, (prev_x, prev_y), (x, y), 1, thickness)
            prev_x, prev_y = x, y
            
    # Đục vài lỗ nhỏ ngẫu nhiên để không biến thành đường cụt hoàn toàn
    for _ in range(5):
        rx, ry = random.randint(0, size), random.randint(0, size)
        cv2.circle(grid, (rx, ry), 10, 0, -1)
        
    return grid

def create_spider_web(size):
    """Tạo mạng nhện chằng chịt"""
    grid = np.zeros((size, size), dtype=np.uint8)
    
    # Tạo các điểm nút ngẫu nhiên
    num_nodes = random.randint(30, 50)
    nodes = []
    for _ in range(num_nodes):
        nodes.append((random.randint(0, size), random.randint(0, size)))
        
    # Nối các điểm gần nhau
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            p1 = nodes[i]
            p2 = nodes[j]
            dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            
            # Chỉ nối nếu khoảng cách < ngưỡng (để tạo mạng cục bộ)
            if dist < 80:
                thickness = random.randint(1, 3)
                cv2.line(grid, p1, p2, 1, thickness)
                
    return grid

def create_asteroid_field(size):
    """Tạo vành đai tiểu hành tinh (Đa giác lồi ngẫu nhiên)"""
    grid = np.zeros((size, size), dtype=np.uint8)
    
    num_asteroids = random.randint(40, 80)
    
    for _ in range(num_asteroids):
        center = (random.randint(0, size), random.randint(0, size))
        radius = random.randint(5, 15)
        
        # Vẽ đa giác ngẫu nhiên
        num_pts = random.randint(3, 6) # Tam giác đến lục giác
        pts = []
        for i in range(num_pts):
            angle = 2 * math.pi * i / num_pts + random.uniform(-0.5, 0.5)
            r = radius * random.uniform(0.8, 1.2)
            x = int(center[0] + r * math.cos(angle))
            y = int(center[1] + r * math.sin(angle))
            pts.append([x, y])
        
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(grid, [pts], 1)
        
    return grid

def create_barcode(size):
    """Tạo song sắt / Mã vạch (Khe hẹp song song)"""
    grid = np.zeros((size, size), dtype=np.uint8)
    
    # Vẽ các thanh dọc hoặc ngang
    is_vertical = random.choice([True, False])
    
    step = random.randint(20, 40)
    bar_thickness = random.randint(5, 15)
    
    if is_vertical:
        for x in range(20, size-20, step):
            # Vẽ thanh dọc
            cv2.rectangle(grid, (x, 0), (x+bar_thickness, size), 1, -1)
            # Đục 1-2 cửa ngẫu nhiên trên mỗi thanh
            for _ in range(random.randint(1, 2)):
                door_y = random.randint(20, size-20)
                door_h = random.randint(15, 30) # Cửa hẹp
                cv2.rectangle(grid, (x, door_y), (x+bar_thickness, door_y+door_h), 0, -1)
    else:
        for y in range(20, size-20, step):
            cv2.rectangle(grid, (0, y), (size, y+bar_thickness), 1, -1)
            for _ in range(random.randint(1, 2)):
                door_x = random.randint(20, size-20)
                door_w = random.randint(15, 30)
                cv2.rectangle(grid, (door_x, y), (door_x+door_w, y+bar_thickness), 0, -1)
                
    return grid

# --- MAIN GENERATOR ---
def generate_chaos_sample(index):
    rand = random.random()
    if rand < 0.25:
        grid_map = create_spiral(MAP_SIZE)
        m_type = "Spiral (Xoan oc)"
    elif rand < 0.5:
        grid_map = create_spider_web(MAP_SIZE)
        m_type = "Spider Web (Mang nhen)"
    elif rand < 0.75:
        grid_map = create_asteroid_field(MAP_SIZE)
        m_type = "Asteroids (Tieu hanh tinh)"
    else:
        grid_map = create_barcode(MAP_SIZE)
        m_type = "Barcode (Song sat)"

    # Tìm Start/Goal
    path = None
    attempts = 0
    while attempts < 200: # Tăng số lần thử vì map này khó tìm đường
        sx, sy = random.randint(5, MAP_SIZE-5), random.randint(5, MAP_SIZE-5)
        gx, gy = random.randint(5, MAP_SIZE-5), random.randint(5, MAP_SIZE-5)
        
        if grid_map[sy, sx] == 0 and grid_map[gy, gx] == 0:
            dist = np.sqrt((sx-gx)**2 + (sy-gy)**2)
            # Với Spiral, khoảng cách ngắn nhưng đường đi thực tế rất dài
            if dist > 80: 
                path = a_star_search(grid_map, (sx, sy), (gx, gy))
                if path: break
        attempts += 1
        
    if path is None: return False

    # Label & Save
    label_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    for (px, py) in path: label_map[py, px] = 1.0
    label_map = cv2.dilate(label_map, np.ones((3,3), np.uint8), iterations=1)
    
    start_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    cv2.circle(start_map, (sx, sy), 4, 1.0, -1)
    goal_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    cv2.circle(goal_map, (gx, gy), 4, 1.0, -1)
    
    input_data = np.stack([grid_map.astype(np.float32), start_map, goal_map], axis=-1)
    np.save(f"{DATA_DIR}/inputs/{index}.npy", input_data)
    np.save(f"{DATA_DIR}/labels/{index}.npy", label_map)

    # Preview
    if SHOW_PREVIEW:
        vis = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        
        # Color Style: "Matrix" (Green & Black) hoặc "Sci-fi"
        vis[grid_map==1] = [0, 0, 0]        # Vật cản đen
        vis[grid_map==0] = [30, 30, 30]     # Nền xám
        
        # Vẽ vật cản nổi bật
        if m_type.startswith("Asteroids"):
            vis[grid_map==1] = [100, 100, 150] # Đá màu xám xanh
        elif m_type.startswith("Spiral"):
            vis[grid_map==1] = [50, 0, 100]    # Tường tím
        elif m_type.startswith("Spider"):
            vis[grid_map==1] = [200, 200, 200] # Tơ trắng
        else: # Barcode
            vis[grid_map==1] = [0, 50, 100]    # Song sắt nâu
            
        vis[label_map > 0] = [0, 255, 255] # Đường đi Vàng
        cv2.circle(vis, (sx, sy), 5, (0, 255, 0), -1)
        cv2.circle(vis, (gx, gy), 5, (0, 0, 255), -1)
        
        cv2.putText(vis, m_type, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.imshow("Chaos Generator", vis)
        cv2.waitKey(50)

    return True

if __name__ == "__main__":
    print(f"Đang tạo {NUM_SAMPLES} bản đồ Hỗn mang (Chaos)...")
    count = 0
    while count < NUM_SAMPLES:
        if generate_chaos_sample(count):
            print(f" -> Chaos {count+1} OK")
            count += 1
    cv2.destroyAllWindows()