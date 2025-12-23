import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import os

class Node:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, occupancy_grid, start, goal, max_iter=100000, step_size=3, search_radius=8):
        self.grid = occupancy_grid
        self.height, self.width = occupancy_grid.shape
        self.start = Node(*self.clamp_to_free(start[0], start[1]))
        self.goal = Node(*self.clamp_to_free(goal[0], goal[1]))
        self.max_iter = max_iter
        self.step_size = step_size
        self.search_radius = search_radius
        self.node_list = [self.start]

    def clamp_to_free(self, x, y):
        for r in range(10):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    nx, ny = int(x+dx), int(y+dy)
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if self.grid[ny, nx] < 0.5: return nx, ny
        return x, y

    def is_collision(self, n1, n2):
        dist = math.hypot(n2.x - n1.x, n2.y - n1.y)
        steps = int(dist) + 1
        line_pts = np.linspace([n1.x, n1.y], [n2.x, n2.y], num=steps)
        for pt in line_pts:
            ix, iy = int(pt[0]), int(pt[1])
            if 0 <= ix < self.width and 0 <= iy < self.height:
                if self.grid[iy, ix] > 0.5: return True
            else: return True
        return False

    def plan(self):
        print(f"--- Đang tìm đường (Max: {self.max_iter} vòng) ---")
        for i in range(self.max_iter):
            if random.random() < 0.02: 
                rx, ry = self.goal.x, self.goal.y
            else:
                rx, ry = random.randint(0, self.width-1), random.randint(0, self.height-1)

            nearest = self.node_list[0]
            min_d = (rx - nearest.x)**2 + (ry - nearest.y)**2
            for n in self.node_list:
                d = (rx - n.x)**2 + (ry - n.y)**2
                if d < min_d:
                    min_d, nearest = d, n
            
            theta = math.atan2(ry - nearest.y, rx - nearest.x)
            new_node = Node(int(nearest.x + self.step_size * math.cos(theta)),
                            int(nearest.y + self.step_size * math.sin(theta)))

            if 0 <= new_node.x < self.width and 0 <= new_node.y < self.height:
                if not self.is_collision(nearest, new_node):
                    new_node.parent = nearest
                    new_node.cost = nearest.cost + self.step_size
                    self.node_list.append(new_node)
                    if math.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) < self.step_size * 2:
                        self.goal.parent = new_node
                        print(f"THÀNH CÔNG tại vòng lặp: {i}")
                        return self.extract_path()
            if i % 10000 == 0 and i > 0:
                print(f"Đã thử {i} vòng...")
        return None

    def extract_path(self):
        path = []
        n = self.goal
        while n:
            path.append([n.x, n.y])
            n = n.parent
        return path[::-1]

    def smooth_path(self, path):
        if not path or len(path) < 3: return path
        smoothed = [path[0]]
        curr = 0
        while curr < len(path) - 1:
            best_next = curr + 1
            for next_idx in range(len(path)-1, curr + 1, -1):
                if not self.is_collision(Node(*path[curr]), Node(*path[next_idx])):
                    best_next = next_idx
                    break
            smoothed.append(path[best_next])
            curr = best_next
        return smoothed

# --- PHẦN DỄ XÀI: CHỈ CẦN NHẬP ĐƯỜNG DẪN FILE ---
if __name__ == "__main__":
    # BẠN CHỈ CẦN SỬA DÒNG NÀY (Copy path từ VS Code dán vào đây)
    input_file = "dataset_complex/labels/98.npy" 
    
    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file tại {input_file}")
    else:
        # Tự động tìm file label tương ứng
        label_file = input_file.replace("inputs", "labels")
        data = np.load(input_file)
        
        # Tách Start/Goal tự động
        if data.ndim == 3: # Nếu file có sẵn 3 lớp (Urban, Pro...)
            obs_map = data[:,:,0]
            sy, sx = np.where(data[:,:,1] > 0.5)
            gy, gx = np.where(data[:,:,2] > 0.5)
            start_pos, goal_pos = (sx[0], sy[0]), (gx[0], gy[0])
        else: # Nếu file 2D (Special...), lấy từ file label
            obs_map = data
            label_data = np.load(label_file)
            ly, lx = np.where(label_data > 0)
            start_pos, goal_pos = (lx[0], ly[0]), (lx[-1], ly[-1])

        # Chạy thuật toán và tính thời gian
        rrt = RRTStar(obs_map, start_pos, goal_pos)
        print(f"Đang xử lý: {input_file}")
        
        start_time = time.time()
        raw_path = rrt.plan()
        end_time = time.time()
        exec_time = end_time - start_time

        # Hiển thị kết quả
        plt.imshow(obs_map, cmap='gray_r')
        plt.scatter(start_pos[0], start_pos[1], c='g', s=100, label='Start')
        plt.scatter(goal_pos[0], goal_pos[1], c='b', s=100, label='Goal')

        if raw_path:
            smoothed = rrt.smooth_path(raw_path)
            raw_path, smoothed = np.array(raw_path), np.array(smoothed)
            path_len = sum(math.hypot(smoothed[i+1][0]-smoothed[i][0], smoothed[i+1][1]-smoothed[i][1]) for i in range(len(smoothed)-1))
            
            plt.plot(raw_path[:, 0], raw_path[:, 1], 'r--', alpha=0.3)
            plt.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=3, label='Path Found')
            
            print(f"\n" + "="*30)
            print(f"THỜI GIAN CHẠY: {exec_time:.4f} giây")
            print(f"ĐỘ DÀI ĐƯỜNG:   {path_len:.2f} pixels")
            print(f"TỔNG SỐ NODE:   {len(rrt.node_list)}")
            print("="*30 + "\n")
            plt.title(f"Success! Time: {exec_time:.2f}s")
        else:
            print(f"Thất bại sau {exec_time:.2f}s")
            plt.title("Fail!")

        plt.legend()
        plt.show()