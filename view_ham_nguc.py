import numpy as np
import cv2
import os
import random

# Lấy ngẫu nhiên
files = os.listdir("./dataset_ham_nguc/inputs")
if not files: exit()

IDX = random.choice(files).split('.')[0]

inp = np.load(f"./dataset_ham_nguc/inputs/50.npy")
lbl = np.load(f"./dataset_ham_nguc/labels/50.npy")

obstacles = inp[:, :, 0]
start_pt = inp[:, :, 1]
goal_pt = inp[:, :, 2]

h, w = obstacles.shape
vis = np.zeros((h, w, 3), dtype=np.uint8)

# --- THEME: UNDERGROUND MINE ---
# 1. Nền đường đi (Đen xám bụi bặm)
vis[:] = [30, 30, 35] 

# 2. Tường (Nâu đất đá sần sùi)
# Tạo texture nhiễu nhẹ cho tường
noise = np.random.randint(0, 50, (h, w), dtype=np.uint8)
wall_color = np.zeros((h, w, 3), dtype=np.uint8)
wall_color[:] = [40, 80, 120] # Màu nâu (BGR)
wall_color = cv2.add(wall_color, np.dstack([noise, noise, noise])) # Thêm nhiễu
vis[obstacles == 1] = wall_color[obstacles == 1]

# 3. Đường đi (Màu Vàng cam - Torch Light)
path_mask = (lbl > 0).astype(np.uint8)
vis[path_mask == 1] = [50, 200, 255] # Vàng sáng
# Glow effect cho đường đi
glow = cv2.GaussianBlur(vis, (15, 15), 0)
vis = cv2.addWeighted(vis, 0.7, glow, 0.3, 0)

# 4. Start / Goal
sy, sx = np.where(start_pt == 1)
gy, gx = np.where(goal_pt == 1)

if len(sy) > 0:
    cv2.circle(vis, (sx[0], sy[0]), 6, (255, 0, 0), -1) # Blue
    cv2.putText(vis, "START", (sx[0]-15, sy[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

if len(gy) > 0:
    cv2.circle(vis, (gx[0], gy[0]), 6, (0, 0, 255), -1) # Red
    cv2.putText(vis, "GOAL", (gx[0]-15, gy[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# Phóng to
vis_large = cv2.resize(vis, (w*3, h*3), interpolation=cv2.INTER_NEAREST)
cv2.imshow("Dungeon Explorer", vis_large)
cv2.waitKey(0)
cv2.destroyAllWindows()