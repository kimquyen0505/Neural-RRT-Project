import numpy as np
import cv2
import os
import heapq
import random
import math

# --- CẤU HÌNH ---
MAP_SIZE = 256
NUM_SAMPLES = 500
DATA_DIR = "./dataset_cau_truc"
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
        if iters > 80000: return None

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

# --- CÁC THUẬT TOÁN ĐỊA HÌNH KỲ DỊ ---

def create_pcb_layout(size):
    """Tạo map kiểu bo mạch điện tử (PCB)"""
    grid = np.zeros((size, size), dtype=np.uint8)
    
    num_traces = random.randint(20, 40)
    
    for _ in range(num_traces):
        # Điểm đầu và cuối
        x1, y1 = random.randint(10, size-10), random.randint(10, size-10)
        x2, y2 = random.randint(10, size-10), random.randint(10, size-10)
        thickness = random.randint(2, 5)
        
        # Vẽ chân linh kiện (Pads) - Tròn hoặc vuông
        cv2.circle(grid, (x1, y1), thickness+2, 1, -1)
        cv2.circle(grid, (x2, y2), thickness+2, 1, -1)
        
        # Vẽ đường mạch đi kiểu Manhattan hoặc 45 độ
        if random.random() < 0.5:
            # Kiểu vuông góc (L-shape)
            mid_x = x2
            cv2.line(grid, (x1, y1), (mid_x, y1), 1, thickness)
            cv2.line(grid, (mid_x, y1), (x2, y2), 1, thickness)
        else:
            # Kiểu trực tiếp
            cv2.line(grid, (x1, y1), (x2, y2), 1, thickness)
            
    # Thêm vài con chip (IC) hình chữ nhật đen chặn đường
    for _ in range(random.randint(3, 6)):
        w, h = random.randint(20, 50), random.randint(20, 50)
        x = random.randint(0, size-w)
        y = random.randint(0, size-h)
        cv2.rectangle(grid, (x, y), (x+w, y+h), 1, -1)
        
    return grid

def create_voronoi_cells(size):
    """Tạo map dạng tế bào Voronoi"""
    grid = np.zeros((size, size), dtype=np.uint8)
    
    # Tạo các điểm hạt giống (Seeds)
    subdiv = cv2.Subdiv2D((0, 0, size, size))
    num_seeds = random.randint(30, 60)
    
    for _ in range(num_seeds):
        pt = (random.randint(0, size-1), random.randint(0, size-1))
        subdiv.insert(pt)
        
    # Lấy danh sách các cạnh Voronoi
    (facets, centers) = subdiv.getVoronoiFacetList([])
    
    # Vẽ các cạnh thành tường
    for i in range(len(facets)):
        facet = facets[i]
        facet_pts = []
        for f in facet:
            facet_pts.append([int(f[0]), int(f[1])])
        
        facet_pts = np.array(facet_pts, np.int32)
        
        # Vẽ viền đa giác (Tường)
        cv2.polylines(grid, [facet_pts], True, 1, thickness=random.randint(2, 4))
        
        # Đục lỗ ngẫu nhiên trên cạnh để thông nhau
        if len(facet_pts) > 1:
            pt1 = facet_pts[0]
            pt2 = facet_pts[1] # Lấy đại 1 cạnh
            mid_pt = ((pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2)
            cv2.circle(grid, mid_pt, 4, 0, -1)
            
    return grid

def create_archipelago_noise(size):
    """Tạo quần đảo bằng nhiễu mờ (Simulated Perlin)"""
    # Tạo noise độ phân giải thấp
    low_res_size = 16 
    noise = np.random.randint(0, 255, (low_res_size, low_res_size), dtype=np.uint8)
    
    # Phóng to lên bằng Cubic Interpolation để làm mịn (tạo đồi núi trập trùng)
    noise_smooth = cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
    
    # Threshold để tạo đảo: > 120 là đất (vật cản), < 120 là nước (đường đi)
    # Hoặc ngược lại tùy ý. Ở đây: 1 là vật cản.
    thresh_val = random.randint(100, 160)
    _, grid = cv2.threshold(noise_smooth, thresh_val, 1, cv2.THRESH_BINARY)
    
    return grid

def create_radial_fortress(size):
    """Tạo pháo đài vòng tròn đồng tâm"""
    grid = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size//2, size//2
    
    num_rings = random.randint(4, 7)
    spacing = (size // 2) // num_rings
    
    # Vẽ các vòng tròn
    for i in range(1, num_rings):
        radius = i * spacing
        cv2.circle(grid, (cx, cy), radius, 1, thickness=random.randint(3, 6))
        
        # Đục lỗ (Cổng thành) trên mỗi vòng
        num_gates = random.randint(2, 5)
        for _ in range(num_gates):
            angle = random.uniform(0, 2*math.pi)
            gate_x = int(cx + radius * math.cos(angle))
            gate_y = int(cy + radius * math.sin(angle))
            cv2.circle(grid, (gate_x, gate_y), random.randint(8, 15), 0, -1)
            
    # Vẽ các nan hoa (Spokes) chắn đường
    num_spokes = random.randint(3, 8)
    for i in range(num_spokes):
        angle = (2 * math.pi / num_spokes) * i + random.uniform(-0.2, 0.2)
        end_x = int(cx + (size//2) * math.cos(angle))
        end_y = int(cy + (size//2) * math.sin(angle))
        
        # Vẽ tia, nhưng chừa khoảng trống ở tâm
        start_x = int(cx + 30 * math.cos(angle))
        start_y = int(cy + 30 * math.sin(angle))
        cv2.line(grid, (start_x, start_y), (end_x, end_y), 1, thickness=3)
        
        # Đục lỗ trên tia
        gap_dist = random.randint(50, size//2 - 20)
        gap_x = int(cx + gap_dist * math.cos(angle))
        gap_y = int(cy + gap_dist * math.sin(angle))
        cv2.circle(grid, (gap_x, gap_y), 10, 0, -1)
        
    return grid

# --- GENERATOR CHÍNH ---
def generate_exotic_sample(index):
    rand = random.random()
    if rand < 0.25:
        grid_map = create_pcb_layout(MAP_SIZE)
        m_type = "PCB (Bo mach)"
    elif rand < 0.5:
        grid_map = create_voronoi_cells(MAP_SIZE)
        m_type = "Voronoi (Te bao)"
    elif rand < 0.75:
        grid_map = create_archipelago_noise(MAP_SIZE)
        m_type = "Archipelago (Quan dao)"
    else:
        grid_map = create_radial_fortress(MAP_SIZE)
        m_type = "Radial (Phao dai)"

    # Tìm Start/Goal
    path = None
    attempts = 0
    while attempts < 150:
        sx, sy = random.randint(5, MAP_SIZE-5), random.randint(5, MAP_SIZE-5)
        gx, gy = random.randint(5, MAP_SIZE-5), random.randint(5, MAP_SIZE-5)
        
        if grid_map[sy, sx] == 0 and grid_map[gy, gx] == 0:
            dist = np.sqrt((sx-gx)**2 + (sy-gy)**2)
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
        
        if "PCB" in m_type:
            vis[:] = [0, 50, 0] # Nền xanh mạch
            vis[grid_map==1] = [0, 180, 0] # Đường mạch sáng
        elif "Voronoi" in m_type:
            vis[:] = [20, 20, 20]
            vis[grid_map==1] = [100, 100, 255] # Tường đỏ
        elif "Archipelago" in m_type:
            vis[:] = [200, 150, 100] # Nước xanh lơ
            vis[grid_map==1] = [50, 100, 50] # Đảo xanh lá
        else: # Radial
            vis[:] = [30, 30, 40]
            vis[grid_map==1] = [0, 255, 255] # Vàng
            
        vis[label_map > 0] = [0, 0, 255] if "Archipelago" not in m_type else [255, 255, 0]
        
        cv2.circle(vis, (sx, sy), 5, (255, 255, 255), -1)
        cv2.circle(vis, (gx, gy), 5, (255, 0, 0), -1)
        
        cv2.putText(vis, m_type, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.imshow("Exotic Generator", vis)
        cv2.waitKey(50)

    return True

if __name__ == "__main__":
    print(f"Đang tạo {NUM_SAMPLES} bản đồ Exotic...")
    count = 0
    while count < NUM_SAMPLES:
        if generate_exotic_sample(count):
            print(f" -> Exotic {count+1} OK")
            count += 1
    cv2.destroyAllWindows()