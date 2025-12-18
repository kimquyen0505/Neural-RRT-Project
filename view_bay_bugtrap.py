import numpy as np
import cv2
import os
import random

# Lấy ngẫu nhiên file trong folder bugtrap
files = os.listdir("./dataset_bay_bugtrap/inputs")
if not files: exit()

IDX = random.choice(files).split('.')[0]

inp = np.load(f"./dataset_bay_bugtrap/inputs/35.npy")
lbl = np.load(f"./dataset_bay_bugtrap/labels/35.npy")

obstacles = inp[:, :, 0]
start_pt = inp[:, :, 1]
goal_pt = inp[:, :, 2]

# --- VISUALIZATION PRO ---
h, w = obstacles.shape
vis = np.zeros((h, w, 3), dtype=np.uint8)
vis[:] = [30, 30, 30] # Nền xám đậm

# Vật cản (Bẫy) - Màu Cam cháy
vis[obstacles == 1] = [0, 100, 255] 
# Viền vật cản cho nét
edges = cv2.Canny((obstacles*255).astype(np.uint8), 100, 200)
vis[edges > 0] = [255, 255, 255]

# Đường đi - Màu Xanh Neon
vis[lbl > 0] = [0, 255, 200]

# Start / Goal
sy, sx = np.where(start_pt == 1)
gy, gx = np.where(goal_pt == 1)

if len(sy) > 0:
    cv2.circle(vis, (sx[0], sy[0]), 5, (0, 255, 0), -1) # Start Xanh lá
    cv2.putText(vis, "Start", (sx[0]-10, sy[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

if len(gy) > 0:
    cv2.circle(vis, (gx[0], gy[0]), 5, (0, 0, 255), -1) # Goal Đỏ
    cv2.putText(vis, "Goal", (gx[0]-10, gy[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

# Phóng to x3
vis_large = cv2.resize(vis, (w*3, h*3), interpolation=cv2.INTER_NEAREST)
cv2.imshow("Trap Inspection", vis_large)
cv2.waitKey(0)
cv2.destroyAllWindows()