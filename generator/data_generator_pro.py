import numpy as np
import cv2
import os
import heapq
import random

# --- CẤU HÌNH PRO ---
MAP_SIZE = 256          # Tăng độ phân giải lên 256x256
NUM_SAMPLES = 200        # Số lượng mẫu (Chạy thử ít thôi vì map to chạy lâu hơn)
DATA_DIR = "./dataset_pro"
SHOW_PREVIEW = True     # Bật preview để thấy độ đẹp của map

if not os.path.exists(f"{DATA_DIR}/inputs"): os.makedirs(f"{DATA_DIR}/inputs")
if not os.path.exists(f"{DATA_DIR}/labels"): os.makedirs(f"{DATA_DIR}/labels")

# --- THUẬT TOÁN A* (Tối ưu hóa cho map lớn) ---
class Node:
    def __init__(self, x, y, cost=0, h=0, parent=None):
        self.x = x; self.y = y; self.cost = cost; self.h = h; self.parent = parent
    def __lt__(self, other): return (self.cost + self.h) < (other.cost + other.h)

def a_star_search(grid, start, goal):
    rows, cols = grid.shape
    open_list = []
    # Dùng Dictionary để tra cứu closed_set nhanh hơn set thường
    g_score = {} 
    
    start_node = Node(start[0], start[1], 0, np.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2))
    heapq.heappush(open_list, start_node)
    g_score[(start[0], start[1])] = 0
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    while open_list:
        current = heapq.heappop(open_list)
        
        if (current.x, current.y) == goal:
            path = []
            while current:
                path.append((current.x, current.y))
                current = current.parent
            return path[::-1]
        
        # Nếu đã tìm thấy đường đi tốt hơn đến điểm này rồi thì bỏ qua
        if (current.x, current.y) in g_score and g_score[(current.x, current.y)] < current.cost:
            continue

        for dx, dy in directions:
            nx, ny = current.x + dx, current.y + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if grid[ny, nx] == 0: # 0 là đường đi
                    move_cost = 1.414 if dx!=0 and dy!=0 else 1.0
                    new_cost = current.cost + move_cost
                    
                    if (nx, ny) not in g_score or new_cost < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = new_cost
                        h = np.sqrt((nx-goal[0])**2 + (ny-goal[1])**2)
                        # Heuristic weight = 1.2 giúp A* chạy nhanh hơn (Sub-optimal nhưng nhanh)
                        heapq.heappush(open_list, Node(nx, ny, new_cost, h*1.2, current))
    return None

# --- THUẬT TOÁN TẠO MAP "XỊN" ---

def create_cellular_automata_map(size, fill_prob=0.45, iterations=5):
    """Tạo map dạng hang động tự nhiên"""
    # 1. Khởi tạo ngẫu nhiên noise
    grid = (np.random.rand(size, size) < fill_prob).astype(np.uint8)
    
    # 2. Làm mịn (Smooth) bằng quy tắc tế bào
    for _ in range(iterations):
        new_grid = grid.copy()
        for y in range(1, size-1):
            for x in range(1, size-1):
                # Đếm số tường xung quanh ô 3x3
                neighbors = np.sum(grid[y-1:y+2, x-1:x+2]) - grid[y, x]
                if grid[y, x] == 1:
                    new_grid[y, x] = 1 if neighbors >= 4 else 0
                else:
                    new_grid[y, x] = 1 if neighbors >= 5 else 0
        grid = new_grid
        
    # Đóng biên (Tường bao quanh)
    grid[0, :] = 1; grid[-1, :] = 1; grid[:, 0] = 1; grid[:, -1] = 1
    return grid

def create_dungeon_rooms(size):
    """Tạo map dạng phòng ốc nối liền (Warehouse style)"""
    grid = np.ones((size, size), dtype=np.uint8) # Ban đầu toàn tường
    
    num_rooms = random.randint(15, 30)
    rooms = []
    
    for _ in range(num_rooms):
        w = random.randint(15, 40)
        h = random.randint(15, 40)
        x = random.randint(1, size - w - 1)
        y = random.randint(1, size - h - 1)
        
        # Đục phòng rỗng
        grid[y:y+h, x:x+w] = 0
        
        # Lưu tâm phòng để nối hành lang
        center_x = x + w // 2
        center_y = y + h // 2
        
        if len(rooms) > 0:
            # Nối phòng cũ với phòng mới bằng hành lang chữ L
            prev_x, prev_y = rooms[-1]
            # Hàng lang ngang
            if prev_x < center_x: grid[prev_y-2:prev_y+2, prev_x:center_x] = 0
            else: grid[prev_y-2:prev_y+2, center_x:prev_x] = 0
            # Hành lang dọc
            if prev_y < center_y: grid[prev_y:center_y, center_x-2:center_x+2] = 0
            else: grid[center_y:prev_y, center_x-2:center_x+2] = 0
            
        rooms.append((center_x, center_y))
        
    return grid

# --- MAIN GENERATOR ---
def generate_pro_sample(index):
    # Chọn ngẫu nhiên 1 trong 2 kiểu map
    if random.random() < 0.5:
        grid_map = create_cellular_automata_map(MAP_SIZE)
        map_type = "Cave"
    else:
        grid_map = create_dungeon_rooms(MAP_SIZE)
        map_type = "Dungeon"

    # Chọn Start / Goal
    # Mẹo: Chọn điểm random, nếu trúng tường thì chọn lại ngay
    attempts = 0
    while attempts < 1000:
        sx, sy = random.randint(0, MAP_SIZE-1), random.randint(0, MAP_SIZE-1)
        gx, gy = random.randint(0, MAP_SIZE-1), random.randint(0, MAP_SIZE-1)
        
        if grid_map[sy, sx] == 0 and grid_map[gy, gx] == 0:
            dist = np.sqrt((sx-gx)**2 + (sy-gy)**2)
            if dist > MAP_SIZE * 0.4: # Khoảng cách phải đủ xa (>40% map)
                path = a_star_search(grid_map, (sx, sy), (gx, gy))
                if path:
                    break
        attempts += 1
    
    if attempts >= 1000 or path is None:
        return False # Map lỗi (bị bịt kín), bỏ qua

    # Xử lý Label (Làm đậm đường đi)
    label_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    for (px, py) in path: label_map[py, px] = 1.0
    label_map = cv2.dilate(label_map, np.ones((5,5), np.uint8), iterations=1) # Dày hơn cho map 256

    # Save Data
    start_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    cv2.circle(start_map, (sx, sy), 5, 1.0, -1)
    goal_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    cv2.circle(goal_map, (gx, gy), 5, 1.0, -1)
    
    input_data = np.stack([grid_map.astype(np.float32), start_map, goal_map], axis=-1)
    np.save(f"{DATA_DIR}/inputs/{index}.npy", input_data)
    np.save(f"{DATA_DIR}/labels/{index}.npy", label_map)

    # --- PREVIEW ĐẸP MẮT ---
    if SHOW_PREVIEW:
        # Tạo ảnh màu nền tối (Dark mode)
        vis = np.zeros((MAP_SIZE, MAP_SIZE, 3), np.uint8)
        
        # Tường màu Xanh đậm (Cyberpunk style)
        vis[grid_map == 1] = [50, 50, 50] 
        
        # Đường đi phát sáng (Màu xanh neon)
        vis[label_map > 0] = [0, 255, 128] # BGR
        
        # Start (Vàng), Goal (Tím)
        cv2.circle(vis, (sx, sy), 6, (0, 255, 255), -1)
        cv2.circle(vis, (gx, gy), 6, (255, 0, 255), -1)
        
        # Thêm text info
        cv2.putText(vis, f"Type: {map_type}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Pro Map Generator", vis)
        cv2.waitKey(100) # Chờ 100ms

    return True

if __name__ == "__main__":
    print(f"Đang tạo {NUM_SAMPLES} bản đồ Pro (256x256)...")
    count = 0
    while count < NUM_SAMPLES:
        if generate_pro_sample(count):
            print(f" -> [{count+1}/{NUM_SAMPLES}] Map OK")
            count += 1
    cv2.destroyAllWindows()