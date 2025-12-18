import numpy as np
import cv2
import os
import heapq
import random

# --- CẤU HÌNH ---
MAP_SIZE = 256          
NUM_SAMPLES = 300       # Số lượng mẫu muốn tạo (Test thử 100 trước, sau đó tăng lên 3000)
DATA_DIR = "./dataset"  # Thư mục lưu
SHOW_PREVIEW = False    # Đặt True nếu muốn xem ảnh map nhảy lên màn hình (chạy chậm hơn)

# Tạo thư mục nếu chưa có
if not os.path.exists(f"{DATA_DIR}/inputs"):
    os.makedirs(f"{DATA_DIR}/inputs")
if not os.path.exists(f"{DATA_DIR}/labels"):
    os.makedirs(f"{DATA_DIR}/labels")

# --- CLASS THUẬT TOÁN A* (A-STAR) ---
class Node:
    def __init__(self, x, y, cost=0, heuristic=0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.heuristic = heuristic
        self.parent = parent
    
    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def heuristic(a, b):
    # Khoảng cách Euclidean
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def a_star_search(grid, start, goal):
    # Grid: 0 là đường, 1 là tường
    rows, cols = grid.shape
    open_list = []
    closed_set = set()
    
    start_node = Node(start[0], start[1], 0, heuristic(start, goal))
    heapq.heappush(open_list, start_node)
    
    # Cho phép đi 8 hướng (ngang, dọc, chéo)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    while open_list:
        current = heapq.heappop(open_list)
        
        if (current.x, current.y) == goal:
            path = []
            while current:
                path.append((current.x, current.y))
                current = current.parent
            return path[::-1] # Đảo ngược lại từ Start -> Goal

        if (current.x, current.y) in closed_set:
            continue
        
        closed_set.add((current.x, current.y))

        for dx, dy in directions:
            nx, ny = current.x + dx, current.y + dy

            # Kiểm tra biên và vật cản
            if 0 <= nx < cols and 0 <= ny < rows:
                if grid[ny, nx] == 0 and (nx, ny) not in closed_set:
                    # Chi phí: Đi thẳng = 1, Đi chéo = 1.414
                    move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                    new_cost = current.cost + move_cost
                    
                    neighbor = Node(nx, ny, new_cost, heuristic((nx, ny), goal), current)
                    heapq.heappush(open_list, neighbor)
    
    return None # Không tìm thấy đường

# --- HÀM TẠO DỮ LIỆU ---
def generate_one_sample(index):
    # 1. Tạo Map nền đen (0)
    grid_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    
    # 2. Vẽ vật cản ngẫu nhiên
    num_obstacles = random.randint(10, 20)
    for _ in range(num_obstacles):
        # Random hình tròn
        cx, cy = random.randint(0, MAP_SIZE), random.randint(0, MAP_SIZE)
        r = random.randint(5, 15)
        cv2.circle(grid_map, (cx, cy), r, 1, -1) # 1 là vật cản

        # Random hình chữ nhật (để đa dạng)
        rx, ry = random.randint(0, MAP_SIZE), random.randint(0, MAP_SIZE)
        rw, rh = random.randint(10, 30), random.randint(10, 30)
        cv2.rectangle(grid_map, (rx, ry), (rx+rw, ry+rh), 1, -1)

    # 3. Chọn Start / Goal
    while True:
        # Lưu ý: OpenCV toạ độ là (x, y), numpy là [row, col] -> [y, x]
        sx, sy = random.randint(0, MAP_SIZE-1), random.randint(0, MAP_SIZE-1)
        gx, gy = random.randint(0, MAP_SIZE-1), random.randint(0, MAP_SIZE-1)

        # Điều kiện: Không trùng vật cản, cách nhau ít nhất 50px
        if grid_map[sy, sx] == 0 and grid_map[gy, gx] == 0:
            dist = np.sqrt((sx-gx)**2 + (sy-gy)**2)
            if dist > 50:
                start = (sx, sy)
                goal = (gx, gy)
                break
    
    # 4. Tìm đường A*
    path = a_star_search(grid_map, start, goal)
    
    if path is None:
        return False # Bỏ qua map này nếu ko tìm được đường
    
    # 5. Xử lý dữ liệu đầu ra (Label)
    label_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    for (px, py) in path:
        label_map[py, px] = 1.0
    
    # Làm đậm đường đi (Dilate) để AI dễ học hơn
    kernel = np.ones((3,3), np.uint8)
    label_map = cv2.dilate(label_map, kernel, iterations=1)

    # 6. Đóng gói Input (3 Channels: Map, Start, Goal)
    start_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    cv2.circle(start_map, start, 3, 1.0, -1) # Vẽ điểm start to chút
    
    goal_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    cv2.circle(goal_map, goal, 3, 1.0, -1)   # Vẽ điểm goal to chút
    
    # Input shape: (128, 128, 3)
    # Channel 0: Obstacles, Channel 1: Start, Channel 2: Goal
    input_data = np.stack([grid_map.astype(np.float32), start_map, goal_map], axis=-1)

    # 7. Lưu file
    np.save(f"{DATA_DIR}/inputs/{index}.npy", input_data)
    np.save(f"{DATA_DIR}/labels/{index}.npy", label_map)

    # --- (Optional) Preview trên Mac để kiểm tra ---
    if SHOW_PREVIEW and index % 5 == 0: # 5 hình hiện 1 lần
        preview = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        # Gán màu: Tường màu xám
        preview[grid_map == 1] = [100, 100, 100]
        # Đường đi màu xanh lá
        preview[label_map > 0] = [0, 255, 0]
        # Start: Xanh dương, Goal: Đỏ
        cv2.circle(preview, start, 3, (255, 0, 0), -1)
        cv2.circle(preview, goal, 3, (0, 0, 255), -1)
        
        cv2.imshow("Data Gen Preview", cv2.resize(preview, (400, 400), interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(100) # Dừng 100ms

    return True

# --- MAIN LOOP ---
if __name__ == "__main__":
    print(f"Bắt đầu sinh {NUM_SAMPLES} mẫu dữ liệu...")
    count = 0
    while count < NUM_SAMPLES:
        if generate_one_sample(count):
            print(f" -> Generated sample {count+1}/{NUM_SAMPLES}")
            count += 1
            
    print("Hoàn tất! Kiểm tra folder 'dataset'.")
    cv2.destroyAllWindows()