import numpy as np
import cv2
import os
import heapq
import random
import sys

# --- CẤU HÌNH ---
MAP_SIZE = 256          # Kích thước map
NUM_SAMPLES = 200       # Số lượng yêu cầu: 200
DATA_DIR = "./dataset_maze_only"
SHOW_PREVIEW = True     # Xem trước khi chạy

# Tăng giới hạn đệ quy để vẽ mê cung lớn không bị lỗi
sys.setrecursionlimit(10**6)

if not os.path.exists(f"{DATA_DIR}/inputs"): os.makedirs(f"{DATA_DIR}/inputs")
if not os.path.exists(f"{DATA_DIR}/labels"): os.makedirs(f"{DATA_DIR}/labels")

# --- THUẬT TOÁN A* ---
class Node:
    def __init__(self, x, y, cost=0, h=0, parent=None):
        self.x = x; self.y = y; self.cost = cost; self.h = h; self.parent = parent
    def __lt__(self, other): return (self.cost + self.h) < (other.cost + other.h)

def a_star_search(grid, start, goal):
    rows, cols = grid.shape
    open_list = []
    g_score = {}
    
    # Heuristic weight = 1.0 (Chuẩn A*) để đảm bảo đường đi ngắn nhất trong mê cung
    start_node = Node(start[0], start[1], 0, np.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2))
    heapq.heappush(open_list, start_node)
    g_score[(start[0], start[1])] = 0
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Trong mê cung chỉ nên đi 4 hướng (tránh đi xuyên góc tường)
    
    while open_list:
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
                        heapq.heappush(open_list, Node(nx, ny, new_cost, h, current))
    return None

# --- THUẬT TOÁN VẼ MÊ CUNG ---
def create_maze_map(size):
    # Kích thước lưới mê cung (nhỏ hơn kích thước ảnh thật)
    # 256 / 8 = 32 ô. Mê cung sẽ là lưới 32x32
    cell_size = 8
    cols, rows = size // cell_size, size // cell_size
    
    # Khởi tạo lưới đầy tường (1)
    maze = np.ones((rows, cols), dtype=np.uint8)
    
    # Hàm đệ quy đào đường
    def carve(cx, cy):
        maze[cy, cx] = 0 # Đánh dấu là đường đi
        
        # Các hướng di chuyển: Lên, Xuống, Trái, Phải
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        random.shuffle(directions) # Xáo trộn ngẫu nhiên
        
        for dx, dy in directions:
            # Nhảy 2 bước để giữ lại tường ngăn cách
            nx, ny = cx + dx*2, cy + dy*2
            
            if 0 <= nx < cols and 0 <= ny < rows:
                if maze[ny, nx] == 1: # Nếu ô đó chưa được đào
                    # Đập thông tường ở giữa ô hiện tại và ô đích
                    maze[cy + dy, cx + dx] = 0
                    carve(nx, ny) # Đệ quy tiếp

    # Bắt đầu đào từ ô (1, 1)
    carve(1, 1)
    
    # Phóng to mê cung lên kích thước thật 256x256
    full_maze = cv2.resize(maze, (size, size), interpolation=cv2.INTER_NEAREST)
    
    # Tạo viền bao quanh (để tránh lỗi ra khỏi map)
    full_maze[0,:]=1; full_maze[-1,:]=1; full_maze[:,0]=1; full_maze[:,-1]=1
    
    return full_maze

# --- HÀM TẠO 1 MẪU ---
def generate_one_maze(index):
    # 1. Tạo Map Mê cung
    grid_map = create_maze_map(MAP_SIZE)
    
    # 2. Chọn Start / Goal
    # Phải loop tìm điểm nằm trên đường (giá trị 0)
    path = None
    attempts = 0
    while attempts < 100:
        sx, sy = random.randint(1, MAP_SIZE-2), random.randint(1, MAP_SIZE-2)
        gx, gy = random.randint(1, MAP_SIZE-2), random.randint(1, MAP_SIZE-2)
        
        # Chỉ chọn nếu cả 2 điểm đều không phải là tường
        if grid_map[sy, sx] == 0 and grid_map[gy, gx] == 0:
            dist = np.sqrt((sx-gx)**2 + (sy-gy)**2)
            if dist > 120: # Start và Goal phải cách xa nhau (>120px)
                # Tìm đường
                path = a_star_search(grid_map, (sx, sy), (gx, gy))
                if path:
                    break
        attempts += 1
    
    if path is None: return False # Bỏ qua nếu ko tìm được đường
    
    # 3. Tạo Label
    label_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    for (px, py) in path: label_map[py, px] = 1.0
    # Làm đậm đường đi
    label_map = cv2.dilate(label_map, np.ones((3,3), np.uint8), iterations=1)
    
    # 4. Lưu Data
    start_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    cv2.circle(start_map, (sx, sy), 4, 1.0, -1)
    goal_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    cv2.circle(goal_map, (gx, gy), 4, 1.0, -1)
    
    input_data = np.stack([grid_map.astype(np.float32), start_map, goal_map], axis=-1)
    np.save(f"{DATA_DIR}/inputs/{index}.npy", input_data)
    np.save(f"{DATA_DIR}/labels/{index}.npy", label_map)
    
    # 5. Preview
    if SHOW_PREVIEW:
        vis = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        vis[grid_map == 1] = [50, 50, 50] # Tường xám
        vis[label_map > 0] = [0, 255, 0]  # Đường xanh lá
        cv2.circle(vis, (sx, sy), 5, (255, 0, 0), -1)
        cv2.circle(vis, (gx, gy), 5, (0, 0, 255), -1)
        
        cv2.putText(vis, f"Maze {index}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.imshow("Maze Generator", vis)
        cv2.waitKey(50) # Chờ 50ms

    return True

# --- MAIN ---
if __name__ == "__main__":
    print(f"Đang tạo {NUM_SAMPLES} Mê Cung...")
    count = 0
    while count < NUM_SAMPLES:
        if generate_one_maze(count):
            print(f" -> Maze {count+1} OK")
            count += 1
    cv2.destroyAllWindows()