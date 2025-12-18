import numpy as np
import cv2
import os
import heapq
import random

# --- CẤU HÌNH ---
MAP_SIZE = 256
NUM_SAMPLES = 500       # Tổng cộng 400 map (khoảng 100 map mỗi loại)
DATA_DIR = "./dataset_urban_pro"
SHOW_PREVIEW = True     # Xem trước

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
    
    # Heuristic weight 1.2: Ưu tiên tìm đường nhanh trong môi trường có cấu trúc
    start_node = Node(start[0], start[1], 0, np.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2))
    heapq.heappush(open_list, start_node)
    g_score[(start[0], start[1])] = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    iters = 0
    while open_list:
        iters += 1
        if iters > 60000: return None # Timeout

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

# --- CÁC HÀM TẠO MAP THEO CHỦ ĐỀ ---

def create_warehouse(size):
    """Nhà kho: Kệ hàng dài song song + Hành lang"""
    grid = np.zeros((size, size), dtype=np.uint8)
    
    # Cấu hình kệ
    aisle_width = random.randint(15, 25) # Độ rộng hành lang đi lại
    shelf_width = random.randint(10, 20) # Độ dày kệ hàng
    margin = 20
    
    # Vẽ kệ dọc hoặc ngang
    is_vertical = random.random() < 0.5
    
    if is_vertical:
        for x in range(margin, size - margin, aisle_width + shelf_width):
            # Vẽ kệ dài, nhưng đục lỗ ngẫu nhiên để làm lối tắt
            y = margin
            while y < size - margin:
                segment_len = random.randint(30, 100)
                end_y = min(y + segment_len, size - margin)
                cv2.rectangle(grid, (x, y), (x + shelf_width, end_y), 1, -1)
                y = end_y + random.randint(15, 30) # Khoảng hở (Cross-aisle)
    else:
        for y in range(margin, size - margin, aisle_width + shelf_width):
            x = margin
            while x < size - margin:
                segment_len = random.randint(30, 100)
                end_x = min(x + segment_len, size - margin)
                cv2.rectangle(grid, (x, y), (end_x, y + shelf_width), 1, -1)
                x = end_x + random.randint(15, 30)
                
    return grid

def create_supermarket(size):
    """Siêu thị: Kệ dày đặc + Quầy thu ngân"""
    # Mật độ cao hơn Warehouse
    grid = np.zeros((size, size), dtype=np.uint8)
    
    aisle_width = random.randint(10, 15) # Lối đi hẹp
    shelf_width = random.randint(8, 12)
    margin = 15
    
    # Vẽ kệ hàng (chiếm 80% phía trên map)
    shopping_area_h = int(size * 0.8)
    
    for x in range(margin, size - margin, aisle_width + shelf_width):
        # Kệ trong siêu thị thường liền mạch, ít lỗ hổng
        cv2.rectangle(grid, (x, margin), (x + shelf_width, shopping_area_h), 1, -1)
        
        # Thỉnh thoảng đục 1 lỗ nhỏ
        if random.random() < 0.5:
            cut_y = random.randint(margin + 20, shopping_area_h - 20)
            cv2.rectangle(grid, (x, cut_y), (x + shelf_width, cut_y + 10), 0, -1)

    # Vẽ quầy thu ngân (Checkout Counters) ở phía dưới
    checkout_y = shopping_area_h + 15
    for x in range(margin, size - margin, 30):
        # Vẽ bàn thu ngân
        cv2.rectangle(grid, (x, checkout_y), (x + 10, size - 10), 1, -1)
        
    return grid

def create_open_office(size):
    """Văn phòng: Cụm bàn ghế (Cubicles) + Cây cảnh"""
    grid = np.zeros((size, size), dtype=np.uint8)
    
    # Chia lưới để đặt bàn
    grid_step = 40
    desk_size = 18
    
    for y in range(20, size - 20, grid_step):
        for x in range(20, size - 20, grid_step):
            if random.random() < 0.8: # 80% ô có bàn
                # Vẽ cụm bàn (Hình vuông hoặc chữ L)
                offset_x = random.randint(0, 5)
                offset_y = random.randint(0, 5)
                cv2.rectangle(grid, (x+offset_x, y+offset_y), 
                              (x+offset_x+desk_size, y+offset_y+desk_size), 1, -1)
                
                # Vẽ ghế (chấm tròn nhỏ gần bàn)
                chair_x = x + offset_x + desk_size + 2
                chair_y = y + offset_y + desk_size // 2
                cv2.circle(grid, (chair_x, chair_y), 3, 1, -1)

    # Thêm chậu cây / máy in (Vật cản ngẫu nhiên)
    for _ in range(20):
        cx, cy = random.randint(10, size-10), random.randint(10, size-10)
        if grid[cy, cx] == 0:
            cv2.circle(grid, (cx, cy), random.randint(4, 6), 1, -1)
            
    return grid

def create_city_park(size):
    """Công viên: Hồ nước lớn + Cây cối rải rác"""
    grid = np.zeros((size, size), dtype=np.uint8)
    
    # 1. Vẽ Hồ nước (Vật cản lớn hình dáng tự nhiên)
    num_lakes = random.randint(1, 3)
    for _ in range(num_lakes):
        center = (random.randint(50, size-50), random.randint(50, size-50))
        axes = (random.randint(20, 60), random.randint(20, 40))
        angle = random.randint(0, 180)
        cv2.ellipse(grid, center, axes, angle, 0, 360, 1, -1)
        
    # 2. Vẽ Đài phun nước / Tượng đài (Hình khối ở giữa)
    if random.random() < 0.5:
        cx, cy = size//2, size//2
        cv2.rectangle(grid, (cx-15, cy-15), (cx+15, cy+15), 1, -1)

    # 3. Vẽ cây cối (Nhiều chấm tròn nhỏ rải rác - Noise)
    for _ in range(80):
        tx, ty = random.randint(5, size-5), random.randint(5, size-5)
        # Kiểm tra không vẽ chồng lên hồ nước
        if grid[ty, tx] == 0:
            cv2.circle(grid, (tx, ty), random.randint(2, 5), 1, -1)
            
    return grid

# --- MAIN GENERATOR ---
def generate_urban_pro_sample(index):
    # Chọn ngẫu nhiên 1 trong 4 chủ đề
    theme_id = random.randint(0, 3)
    
    if theme_id == 0:
        grid_map = create_warehouse(MAP_SIZE)
        theme_name = "Warehouse"
    elif theme_id == 1:
        grid_map = create_open_office(MAP_SIZE)
        theme_name = "Office"
    elif theme_id == 2:
        grid_map = create_supermarket(MAP_SIZE)
        theme_name = "Supermarket"
    else:
        grid_map = create_city_park(MAP_SIZE)
        theme_name = "City Park"

    # Tìm Start/Goal
    path = None
    attempts = 0
    while attempts < 100:
        sx, sy = random.randint(5, MAP_SIZE-5), random.randint(5, MAP_SIZE-5)
        gx, gy = random.randint(5, MAP_SIZE-5), random.randint(5, MAP_SIZE-5)
        
        if grid_map[sy, sx] == 0 and grid_map[gy, gx] == 0:
            dist = np.sqrt((sx-gx)**2 + (sy-gy)**2)
            if dist > 80:
                path = a_star_search(grid_map, (sx, sy), (gx, gy))
                if path: break
        attempts += 1
        
    if path is None: return False

    # Label
    label_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    for (px, py) in path: label_map[py, px] = 1.0
    label_map = cv2.dilate(label_map, np.ones((3,3), np.uint8), iterations=1)
    
    # Inputs
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
        
        # Color coding theo theme
        if theme_name == "Warehouse":
            vis[grid_map==1] = [50, 50, 150] # Kệ đỏ đất
            vis[grid_map==0] = [230, 230, 230] # Sàn sáng
        elif theme_name == "Office":
            vis[grid_map==1] = [100, 100, 100] # Bàn xám
            vis[grid_map==0] = [240, 240, 255] # Sàn xanh nhạt
        elif theme_name == "Supermarket":
            vis[grid_map==1] = [0, 100, 0] # Kệ xanh lá đậm
            vis[grid_map==0] = [255, 255, 255]
        else: # City Park
            vis[grid_map==1] = [255, 0, 0] # Hồ nước xanh dương
            vis[grid_map==0] = [50, 150, 50] # Cỏ xanh
            
        # Vẽ đường đi
        path_mask = (label_map > 0).astype(np.uint8)
        vis[path_mask == 1] = [0, 0, 255] # Đường đi đỏ
        
        cv2.circle(vis, (sx, sy), 6, (0, 255, 255), -1) # Start Vàng
        cv2.circle(vis, (gx, gy), 6, (255, 0, 255), -1) # Goal Tím
        
        cv2.putText(vis, theme_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.imshow("Urban Pro Generator", vis)
        cv2.waitKey(50)

    return True

if __name__ == "__main__":
    print(f"Đang tạo {NUM_SAMPLES} bản đồ Urban Pro (4 themes)...")
    count = 0
    while count < NUM_SAMPLES:
        if generate_urban_pro_sample(count):
            print(f" -> [{count+1}/{NUM_SAMPLES}] Map Created.")
            count += 1
    cv2.destroyAllWindows()