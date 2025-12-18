import numpy as np
import cv2
import os

# Chọn file muốn xem
FILE_INDEX = 0
INPUT_FILE = f"./dataset_pro/inputs/2.npy"
LABEL_FILE = f"./dataset_pro/labels/2.npy"

def view_hd():
    if not os.path.exists(INPUT_FILE):
        print("Chưa có data! Hãy chạy data_generator_pro.py trước.")
        return

    # Load data
    inp = np.load(INPUT_FILE)
    lbl = np.load(LABEL_FILE)
    
    obstacles = inp[:, :, 0]
    start_pt = inp[:, :, 1]
    goal_pt = inp[:, :, 2]
    
    h, w = obstacles.shape
    
    # --- TẠO VISUALIZATION ĐẸP ---
    # 1. Nền: Màu xanh đen vũ trụ
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:] = [20, 20, 30] 
    
    # 2. Vật cản: Màu xám thép, có viền
    # Tạo mask tường
    wall_mask = (obstacles == 1).astype(np.uint8)
    vis[wall_mask == 1] = [60, 60, 75]
    
    # Vẽ viền cho tường (Dùng Edge Detection)
    edges = cv2.Canny(wall_mask * 255, 100, 200)
    vis[edges > 0] = [100, 100, 120] # Viền sáng hơn
    
    # 3. Đường đi: Hiệu ứng Neon Green
    # Dùng GaussianBlur để tạo hiệu ứng tỏa sáng (Glow)
    path_mask = (lbl > 0).astype(np.uint8) * 255
    path_layer = np.zeros_like(vis)
    path_layer[path_mask > 0] = [0, 255, 0] # Xanh lá
    
    # Làm mờ để tạo glow
    glow = cv2.GaussianBlur(path_layer, (9, 9), 0)
    vis = cv2.addWeighted(vis, 1.0, glow, 1.5, 0) # Cộng lớp glow vào nền
    
    # Vẽ lại lõi đường đi cho sắc nét
    vis[path_mask > 0] = [50, 255, 50]

    # 4. Start & Goal
    sy, sx = np.where(start_pt == 1)
    gy, gx = np.where(goal_pt == 1)
    
    if len(sy) > 0:
        # Start: Vòng tròn màu Cyan
        cv2.circle(vis, (sx[0], sy[0]), 6, (255, 255, 0), -1)
        cv2.circle(vis, (sx[0], sy[0]), 8, (255, 255, 255), 1) # Viền trắng
        cv2.putText(vis, "S", (sx[0]-4, sy[0]+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    if len(gy) > 0:
        # Goal: Vòng tròn màu Magenta
        cv2.circle(vis, (gx[0], gy[0]), 6, (255, 0, 255), -1)
        cv2.circle(vis, (gx[0], gy[0]), 8, (255, 255, 255), 1)
        cv2.putText(vis, "G", (gx[0]-4, gy[0]+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    # 5. Phóng to để xem cho đã mắt (x2)
    vis_large = cv2.resize(vis, (w*2, h*2), interpolation=cv2.INTER_NEAREST)
    
    print("Đang hiển thị HD map...")
    cv2.imshow("HD Visualization (Pro Dataset)", vis_large)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    view_hd()