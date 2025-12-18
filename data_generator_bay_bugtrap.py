import numpy as np
import cv2
import os
import heapq
import random

# --- CẤU HÌNH ---
MAP_SIZE = 256
NUM_SAMPLES = 200
DATA_DIR = "./dataset_bugtrap"
SHOW_PREVIEW = True

if not os.path.exists(f"{DATA_DIR}/inputs"): os.makedirs(f"{DATA_DIR}/inputs")
if not os.path.exists(f"{DATA_DIR}/labels"): os.makedirs(f"{DATA_DIR}/labels")

# --- A* SOLVER (Weighted để tìm đường thoát bẫy nhanh hơn) ---
class Node:
    def __init__(self, x, y, cost=0, h=0, parent=None):
        self.x = x; self.y = y; self.cost = cost; self.h = h; self.parent = parent
    def __lt__(self, other): return (self.cost + self.h) < (other.cost + other.h)

def a_star_search(grid, start, goal):
    rows, cols = grid.shape
    open_list = []
    g_score = {}
    
    # Heuristic x 1.2 để ưu tiên tìm đường thoát nhanh
    start_node = Node(start[0], start[1], 0, np.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2))
    heapq.heappush(open_list, start_node)
    g_score[(start[0], start[1])] = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    max_steps = 100000 # Giới hạn bước để tránh treo máy nếu bẫy quá kín
    steps = 0
    
    while open_list:
        steps += 1
        if steps > max_steps: return None 

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

# --- CÁC HÀM VẼ BẪY (TRAP PAINTERS) ---

def draw_u_trap(img, x, y, w, h, thickness, rotation):
    """Vẽ chữ U hoặc C xoay các hướng"""
    # Vẽ khối đặc trước
    cv2.rectangle(img, (x, y), (x+w, y+h), 1, -1)
    
    # Khoét rỗng bên trong tùy theo hướng xoay
    # 0: U (Mở trên), 1: U ngược (Mở dưới), 2: C (Mở trái), 3: C ngược (Mở phải)
    inner_x, inner_y = x + thickness, y + thickness
    inner_w, inner_h = w - 2*thickness, h - 2*thickness
    
    if inner_w > 0 and inner_h > 0:
        if rotation == 0: # Mở trên
            cv2.rectangle(img, (inner_x, y), (inner_x + inner_w, inner_y + inner_h), 0, -1)
            return (x + w//2, y + h//2) # Trả về tâm bẫy
        elif rotation == 1: # Mở dưới
            cv2.rectangle(img, (inner_x, inner_y), (inner_x + inner_w, y + h), 0, -1)
            return (x + w//2, y + h//2)
        elif rotation == 2: # Mở trái
            cv2.rectangle(img, (x, inner_y), (inner_x + inner_w, inner_y + inner_h), 0, -1)
            return (x + w//2, y + h//2)
        elif rotation == 3: # Mở phải
            cv2.rectangle(img, (inner_x, inner_y), (x + w, inner_y + inner_h), 0, -1)
            return (x + w//2, y + h//2)
    return None

def draw_wall_trap(img, x, y, w, h, shape_type):
    """Vẽ tường chắn dạng L, T, H, I"""
    if shape_type == 'I': # Tường thẳng đứng hoặc ngang
        if random.random() < 0.5:
            cv2.rectangle(img, (x, y), (x+w, y+random.randint(5, 15)), 1, -1)
        else:
            cv2.rectangle(img, (x, y), (x+random.randint(5, 15), y+h), 1, -1)
            
    elif shape_type == 'L': # Góc vuông
        th = random.randint(10, 20)
        cv2.rectangle(img, (x, y), (x+th, y+h), 1, -1) # Dọc
        cv2.rectangle(img, (x, y+h-th), (x+w, y+h), 1, -1) # Ngang đáy
        return (x + w//2, y + h//2) # Điểm hõm chữ L

    elif shape_type == 'H':
        th = random.randint(10, 20)
        cv2.rectangle(img, (x, y), (x+th, y+h), 1, -1) # Chân trái
        cv2.rectangle(img, (x+w-th, y), (x+w, y+h), 1, -1) # Chân phải
        cv2.rectangle(img, (x, y+h//2-th//2), (x+w, y+h//2+th//2), 1, -1) # Gạch ngang
        return (x + w//2, y + h//4) # Điểm kẹt trong chữ H

    return None

def generate_bugtrap_sample(index):
    grid_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    
    trap_centers = [] # Lưu các điểm "nhạy cảm" bên trong bẫy
    
    # 1. Vẽ 5-8 cái bẫy lớn (U, C, Box)
    for _ in range(random.randint(5, 8)):
        x, y = random.randint(20, MAP_SIZE-70), random.randint(20, MAP_SIZE-70)
        w, h = random.randint(40, 60), random.randint(40, 60)
        thick = random.randint(5, 15)
        rot = random.randint(0, 3)
        
        # Vẽ U-Trap
        center = draw_u_trap(grid_map, x, y, w, h, thick, rot)
        if center: trap_centers.append(center)

    # 2. Vẽ thêm tường rào chắn (L, H, I)
    for _ in range(random.randint(3, 6)):
        x, y = random.randint(10, MAP_SIZE-50), random.randint(10, MAP_SIZE-50)
        w, h = random.randint(30, 80), random.randint(30, 80)
        sh = random.choice(['L', 'H', 'I'])
        center = draw_wall_trap(grid_map, x, y, w, h, sh)
        if center: trap_centers.append(center)
        
    # 3. Chiến thuật chọn Start/Goal: BẮT BUỘC 1 TRONG 2 PHẢI NẰM TRONG BẪY
    # Để ép đường đi phải vòng ra ngoài
    
    path = None
    attempts = 0
    while attempts < 100:
        # 50% cơ hội Goal nằm trong bẫy, 50% random hoàn toàn
        if trap_centers and random.random() < 0.7: 
            # Lấy 1 điểm trong bẫy làm Goal (cộng trừ nhiễu chút xíu)
            tx, ty = random.choice(trap_centers)
            gx = np.clip(tx + random.randint(-5, 5), 0, MAP_SIZE-1)
            gy = np.clip(ty + random.randint(-5, 5), 0, MAP_SIZE-1)
            
            # Start random bên ngoài
            sx, sy = random.randint(0, MAP_SIZE-1), random.randint(0, MAP_SIZE-1)
        else:
            # Random cả 2
            sx, sy = random.randint(0, MAP_SIZE-1), random.randint(0, MAP_SIZE-1)
            gx, gy = random.randint(0, MAP_SIZE-1), random.randint(0, MAP_SIZE-1)
            
        # Kiểm tra tính hợp lệ
        if grid_map[sy, sx] == 0 and grid_map[gy, gx] == 0:
            dist = np.sqrt((sx-gx)**2 + (sy-gy)**2)
            if dist > 80: # Cách nhau đủ xa
                path = a_star_search(grid_map, (sx, sy), (gx, gy))
                if path:
                    start = (sx, sy)
                    goal = (gx, gy)
                    break
        attempts += 1
        
    if path is None: return False

    # 4. Xử lý Label & Save (Giống các bước trước)
    label_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    for (px, py) in path: label_map[py, px] = 1.0
    label_map = cv2.dilate(label_map, np.ones((3,3), np.uint8), iterations=1)
    
    start_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    cv2.circle(start_map, start, 4, 1.0, -1)
    goal_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    cv2.circle(goal_map, goal, 4, 1.0, -1)
    
    input_data = np.stack([grid_map.astype(np.float32), start_map, goal_map], axis=-1)
    np.save(f"{DATA_DIR}/inputs/{index}.npy", input_data)
    np.save(f"{DATA_DIR}/labels/{index}.npy", label_map)

    # 5. Preview
    if SHOW_PREVIEW:
        vis = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        # Bẫy màu Đỏ cam
        vis[grid_map == 1] = [0, 69, 255] # Orange Red
        # Đường thoát màu Xanh lơ
        vis[label_map > 0] = [255, 255, 0] # Cyan
        
        cv2.circle(vis, start, 5, (0, 255, 0), -1) # Start Green
        cv2.circle(vis, goal, 5, (0, 0, 255), -1)  # Goal Red
        
        # Vẽ mũi tên hướng từ Start -> Goal (ảo) để thấy bị chắn
        cv2.arrowedLine(vis, start, goal, (100, 100, 100), 1, tipLength=0.05)
        
        cv2.putText(vis, f"Bug Trap {index}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.imshow("Trap Generator", vis)
        cv2.waitKey(50)

    return True

if __name__ == "__main__":
    print(f"Đang tạo {NUM_SAMPLES} Bug Trap Maps...")
    count = 0
    while count < NUM_SAMPLES:
        if generate_bugtrap_sample(count):
            print(f" -> Trap {count+1} OK")
            count += 1
    cv2.destroyAllWindows()