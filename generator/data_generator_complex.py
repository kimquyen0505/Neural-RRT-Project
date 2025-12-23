import numpy as np
import cv2
import os
import heapq
import random

# --- CẤU HÌNH ---
MAP_SIZE = 128
NUM_SAMPLES = 200      # Tạo thử 200 mẫu trước
DATA_DIR = "./dataset_complex" # Lưu vào thư mục mới để không lẫn với cái cũ
SHOW_PREVIEW = False   # True để xem khi chạy

if not os.path.exists(f"{DATA_DIR}/inputs"):
    os.makedirs(f"{DATA_DIR}/inputs")
if not os.path.exists(f"{DATA_DIR}/labels"):
    os.makedirs(f"{DATA_DIR}/labels")

# --- CLASS A* (GIỮ NGUYÊN) ---
class Node:
    def __init__(self, x, y, cost=0, heuristic=0, parent=None):
        self.x = x; self.y = y; self.cost = cost; self.heuristic = heuristic; self.parent = parent
    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def a_star_search(grid, start, goal):
    rows, cols = grid.shape
    open_list = []
    closed_set = set()
    start_node = Node(start[0], start[1], 0, np.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2))
    heapq.heappush(open_list, start_node)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    while open_list:
        current = heapq.heappop(open_list)
        if (current.x, current.y) == goal:
            path = []
            while current:
                path.append((current.x, current.y))
                current = current.parent
            return path[::-1]
        if (current.x, current.y) in closed_set: continue
        closed_set.add((current.x, current.y))
        for dx, dy in directions:
            nx, ny = current.x + dx, current.y + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if grid[ny, nx] == 0 and (nx, ny) not in closed_set:
                    move_cost = 1.414 if dx!=0 and dy!=0 else 1.0
                    h = np.sqrt((nx-goal[0])**2 + (ny-goal[1])**2)
                    heapq.heappush(open_list, Node(nx, ny, current.cost + move_cost, h, current))
    return None

# --- CÁC HÀM VẼ HÌNH PHỨC TẠP ---

def draw_random_polygon(img):
    """Vẽ đa giác lồi ngẫu nhiên"""
    center = (random.randint(20, MAP_SIZE-20), random.randint(20, MAP_SIZE-20))
    radius = random.randint(10, 25)
    num_points = random.randint(3, 6) # Tam giác đến Lục giác
    
    points = []
    for i in range(num_points):
        # Chia đều góc xoay quanh tâm
        angle_deg = (360 / num_points) * i + random.randint(-15, 15)
        angle_rad = np.deg2rad(angle_deg)
        # Random bán kính một chút để tạo hình méo
        r = radius * random.uniform(0.7, 1.3)
        x = int(center[0] + r * np.cos(angle_rad))
        y = int(center[1] + r * np.sin(angle_rad))
        points.append([x, y])
    
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], 1)

def draw_u_shape(img):
    """Vẽ bẫy chữ U hoặc chữ C (Khoét rỗng hình chữ nhật)"""
    x = random.randint(10, MAP_SIZE-40)
    y = random.randint(10, MAP_SIZE-40)
    w = random.randint(30, 50)
    h = random.randint(30, 50)
    
    # Vẽ khối đặc trước
    cv2.rectangle(img, (x, y), (x+w, y+h), 1, -1)
    
    # Chọn hướng mở của chữ U (Trên, Dưới, Trái, Phải)
    direction = random.choice(['up', 'down', 'left', 'right'])
    thickness = random.randint(5, 10) # Độ dày tường
    
    # Khoét rỗng bên trong (Gán lại bằng 0)
    if direction == 'up':
        cv2.rectangle(img, (x+thickness, y), (x+w-thickness, y+h-thickness), 0, -1)
    elif direction == 'down':
        cv2.rectangle(img, (x+thickness, y+thickness), (x+w-thickness, y+h), 0, -1)
    elif direction == 'left':
        cv2.rectangle(img, (x, y+thickness), (x+w-thickness, y+h-thickness), 0, -1)
    elif direction == 'right':
        cv2.rectangle(img, (x+thickness, y+thickness), (x+w, y+h-thickness), 0, -1)

# --- HÀM TẠO MAP CHÍNH ---
def generate_complex_sample(index):
    grid_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    
    # 1. Random số lượng vật cản
    num_obstacles = random.randint(8, 15)
    
    for _ in range(num_obstacles):
        shape_type = random.choice(['circle', 'rect', 'poly', 'u_shape'])
        
        if shape_type == 'circle':
            cx, cy = random.randint(0, MAP_SIZE), random.randint(0, MAP_SIZE)
            cv2.circle(grid_map, (cx, cy), random.randint(5, 15), 1, -1)
            
        elif shape_type == 'rect':
            rx, ry = random.randint(0, MAP_SIZE), random.randint(0, MAP_SIZE)
            cv2.rectangle(grid_map, (rx, ry), (rx+random.randint(10,30), ry+random.randint(10,30)), 1, -1)
            
        elif shape_type == 'poly':
            draw_random_polygon(grid_map)
            
        elif shape_type == 'u_shape':
            draw_u_shape(grid_map)

    # 2. Thêm "muối tiêu" (Noise) - các chấm nhỏ li ti
    for _ in range(20):
        nx, ny = random.randint(0, MAP_SIZE-1), random.randint(0, MAP_SIZE-1)
        grid_map[ny, nx] = 1 

    # 3. Chọn Start / Goal (Giữ nguyên logic cũ)
    while True:
        sx, sy = random.randint(0, MAP_SIZE-1), random.randint(0, MAP_SIZE-1)
        gx, gy = random.randint(0, MAP_SIZE-1), random.randint(0, MAP_SIZE-1)
        
        if grid_map[sy, sx] == 0 and grid_map[gy, gx] == 0:
            if np.sqrt((sx-gx)**2 + (sy-gy)**2) > 40: # Khoảng cách > 40
                start = (sx, sy)
                goal = (gx, gy)
                break
    
    # 4. Tìm đường A*
    path = a_star_search(grid_map, start, goal)
    if path is None: return False
    
    # 5. Tạo Label
    label_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    for (px, py) in path: label_map[py, px] = 1.0
    label_map = cv2.dilate(label_map, np.ones((3,3), np.uint8), iterations=1)
    
    # 6. Save Data
    start_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    cv2.circle(start_map, start, 3, 1.0, -1)
    goal_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    cv2.circle(goal_map, goal, 3, 1.0, -1)
    
    input_data = np.stack([grid_map.astype(np.float32), start_map, goal_map], axis=-1)
    
    np.save(f"{DATA_DIR}/inputs/{index}.npy", input_data)
    np.save(f"{DATA_DIR}/labels/{index}.npy", label_map)

    if SHOW_PREVIEW and index % 2 == 0:
        vis = np.zeros((MAP_SIZE, MAP_SIZE, 3), np.uint8)
        vis[grid_map==1] = [120, 120, 120] # Tường xám
        vis[label_map>0] = [0, 255, 0]     # Đường xanh
        cv2.circle(vis, start, 3, (255,0,0), -1)
        cv2.circle(vis, goal, 3, (0,0,255), -1)
        cv2.imshow("Complex Map", cv2.resize(vis, (400,400), interpolation=0))
        cv2.waitKey(50)

    return True

# --- MAIN ---
if __name__ == "__main__":
    print(f"Đang tạo {NUM_SAMPLES} bản đồ phức tạp...")
    count = 0
    while count < NUM_SAMPLES:
        if generate_complex_sample(count):
            print(f" -> Generated complex sample {count+1}")
            count += 1
    cv2.destroyAllWindows()