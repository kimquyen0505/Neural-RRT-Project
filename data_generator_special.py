import numpy as np
import cv2
import os
import heapq
import random
import math

# --- CẤU HÌNH ---
MAP_SIZE = 256
NUM_SAMPLES = 500
DATA_DIR = "./dataset_special"
SHOW_PREVIEW = True

if not os.path.exists(f"{DATA_DIR}/inputs"): os.makedirs(f"{DATA_DIR}/inputs")
if not os.path.exists(f"{DATA_DIR}/labels"): os.makedirs(f"{DATA_DIR}/labels")

# --- A* SOLVER ---
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
                        heapq.heappush(open_list, Node(nx, ny, new_cost, h*1.2, current))
    return None

# --- CÁC THUẬT TOÁN ĐẶC BIỆT ---

def create_parking_lot(size):
    """Tạo bãi đỗ xe"""
    grid = np.zeros((size, size), dtype=np.uint8)
    
    # Chia thành các dãy xe
    lane_width = random.randint(30, 45) # Làn đường xe chạy
    spot_width = random.randint(15, 20) # Chiều rộng 1 chỗ đỗ
    spot_depth = random.randint(30, 40) # Chiều sâu chỗ đỗ (chiều dài xe)
    
    # Vẽ các block đỗ xe
    # Một block gồm 2 hàng xe đấu lưng vào nhau
    block_total_width = spot_depth * 2
    
    x = lane_width // 2
    while x < size - lane_width:
        # Vẽ dải phân cách giữa 2 hàng xe (nếu có)
        # cv2.line(grid, (x + spot_depth, 0), (x + spot_depth, size), 1, 2)
        
        # Vẽ xe
        for y in range(lane_width // 2, size - lane_width // 2, spot_width):
            # 85% xác suất có xe đỗ -> thành vật cản
            if random.random() < 0.85:
                # Xe hàng bên trái
                padding = 2
                cv2.rectangle(grid, (x + padding, y + padding), 
                              (x + spot_depth - padding, y + spot_width - padding), 1, -1)
                
            if random.random() < 0.85:
                # Xe hàng bên phải (đấu lưng)
                padding = 2
                cv2.rectangle(grid, (x + spot_depth + padding, y + padding), 
                              (x + block_total_width - padding, y + spot_width - padding), 1, -1)
        
        # Nhảy qua block xe và làn đường tiếp theo
        x += block_total_width + lane_width
        
    # Tạo biên
    grid[0:5, :] = 1; grid[-5:, :] = 1; grid[:, 0:5] = 1; grid[:, -5:] = 1
    return grid

def create_switchback(size):
    """Tạo đường đèo dốc Zigzag (Canyon)"""
    grid = np.ones((size, size), dtype=np.uint8) # Ban đầu toàn núi (1)
    
    # Tạo đường dẫn hình sin hoặc zigzag
    path_width = random.randint(15, 25)
    amplitude = random.randint(40, 80) # Độ rộng lắc lư
    frequency = random.uniform(0.02, 0.05)
    
    # Chọn hướng: Dọc hay Ngang
    is_vertical = random.random() < 0.5
    
    if is_vertical:
        center_x = size // 2
        for y in range(size):
            # Tính toán x trung tâm của con đường tại y
            offset = int(amplitude * math.sin(y * frequency))
            path_x = center_x + offset
            
            # Đục lỗ (tạo đường)
            start_x = max(0, path_x - path_width)
            end_x = min(size, path_x + path_width)
            grid[y, start_x:end_x] = 0
    else:
        center_y = size // 2
        for x in range(size):
            offset = int(amplitude * math.sin(x * frequency))
            path_y = center_y + offset
            
            start_y = max(0, path_y - path_width)
            end_y = min(size, path_y + path_width)
            grid[start_y:end_y, x] = 0
            
    # Thêm nhiễu đá rơi trên đường đèo
    for _ in range(50):
        rx, ry = random.randint(0, size-1), random.randint(0, size-1)
        if grid[ry, rx] == 0: # Chỉ thả đá xuống đường
            cv2.circle(grid, (rx, ry), random.randint(2, 5), 1, -1)
            
    return grid

def create_swiss_cheese(size):
    """Tạo địa hình Phô mai thủng lỗ (Đặc -> Rỗng)"""
    # Ban đầu là khối đặc (1)
    grid = np.ones((size, size), dtype=np.uint8)
    
    # Đục thật nhiều lỗ tròn chồng lấn lên nhau
    num_holes = random.randint(150, 300)
    for _ in range(num_holes):
        cx, cy = random.randint(0, size), random.randint(0, size)
        radius = random.randint(10, 25)
        cv2.circle(grid, (cx, cy), radius, 0, -1)
        
    return grid

def create_lattice(size):
    """Tạo lưới mắt cáo (Lattice/Diamond Grid)"""
    grid = np.zeros((size, size), dtype=np.uint8)
    
    thickness = random.randint(3, 8)
    spacing = random.randint(30, 50)
    
    # Vẽ các đường chéo thứ nhất (y = x + c)
    for i in range(-size, size, spacing):
        pt1 = (0, i)
        pt2 = (size, i + size)
        cv2.line(grid, pt1, pt2, 1, thickness)
        
    # Vẽ các đường chéo thứ hai (y = -x + c)
    for i in range(0, 2 * size, spacing):
        pt1 = (0, i)
        pt2 = (size, i - size)
        cv2.line(grid, pt1, pt2, 1, thickness)
        
    # Đục lỗ ngẫu nhiên tại các giao điểm để mở đường
    for _ in range(30):
        rx, ry = random.randint(0, size), random.randint(0, size)
        cv2.circle(grid, (rx, ry), 15, 0, -1)
        
    return grid

# --- GENERATOR CHÍNH ---
def generate_special_sample(index):
    rand = random.random()
    if rand < 0.25:
        grid_map = create_parking_lot(MAP_SIZE)
        m_type = "Parking Lot"
    elif rand < 0.5:
        grid_map = create_switchback(MAP_SIZE)
        m_type = "Switchback (Deo)"
    elif rand < 0.75:
        grid_map = create_swiss_cheese(MAP_SIZE)
        m_type = "Swiss Cheese"
    else:
        grid_map = create_lattice(MAP_SIZE)
        m_type = "Lattice (Luoi)"

    # Tìm Start/Goal
    path = None
    attempts = 0
    while attempts < 200:
        sx, sy = random.randint(5, MAP_SIZE-5), random.randint(5, MAP_SIZE-5)
        gx, gy = random.randint(5, MAP_SIZE-5), random.randint(5, MAP_SIZE-5)
        
        if grid_map[sy, sx] == 0 and grid_map[gy, gx] == 0:
            dist = np.sqrt((sx-gx)**2 + (sy-gy)**2)
            # Với Swiss Cheese và Lattice, cần khoảng cách đủ xa để test luồn lách
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
    cv2.circle(start_map, (sx, sy), 5, 1.0, -1)
    goal_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    cv2.circle(goal_map, (gx, gy), 5, 1.0, -1)
    
    input_data = np.stack([grid_map.astype(np.float32), start_map, goal_map], axis=-1)
    np.save(f"{DATA_DIR}/inputs/{index}.npy", input_data)
    np.save(f"{DATA_DIR}/labels/{index}.npy", label_map)

    # Preview
    if SHOW_PREVIEW:
        vis = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        
        # Style Kỹ thuật (Blueprint)
        vis[:] = [100, 50, 0] # Nền Xanh đậm (Blueprint style)
        vis[grid_map==1] = [255, 255, 255] # Vật cản Trắng
        
        if "Cheese" in m_type:
             vis[:] = [30, 30, 30]
             vis[grid_map==1] = [50, 100, 150] # Đá
        elif "Parking" in m_type:
             vis[:] = [50, 50, 50] # Nhựa đường
             vis[grid_map==1] = [200, 200, 200] # Xe ô tô
            
        vis[label_map > 0] = [0, 255, 0] # Đường đi Xanh lá
        
        cv2.circle(vis, (sx, sy), 5, (0, 0, 255), -1)
        cv2.circle(vis, (gx, gy), 5, (0, 255, 255), -1)
        
        cv2.putText(vis, m_type, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.imshow("Special Map Generator", vis)
        cv2.waitKey(50)

    return True

if __name__ == "__main__":
    print(f"Đang tạo {NUM_SAMPLES} bản đồ Đặc thù (Special)...")
    count = 0
    while count < NUM_SAMPLES:
        if generate_special_sample(count):
            print(f" -> Special {count+1} OK")
            count += 1
    cv2.destroyAllWindows()