import numpy as np
import cv2
import os
import random

files = os.listdir("./dataset_hinh_hoc/inputs")
if not files: exit()

IDX = random.choice(files).split('.')[0]

inp = np.load(f"./dataset_hinh_hoc/inputs/7.npy")
lbl = np.load(f"./dataset_hinh_hoc/labels/7.npy")

obstacles = inp[:, :, 0]
start_pt = inp[:, :, 1]
goal_pt = inp[:, :, 2]

h, w = obstacles.shape
vis = np.zeros((h, w, 3), dtype=np.uint8)

# Nền vũ trụ (Đen sâu thẳm)
vis[:] = [10, 10, 20] 

# Vật cản: Hiệu ứng Neon Cyberpunk
# Màu hồng tím (Magenta/Purple)
vis[obstacles == 1] = [128, 0, 128] 
# Viền sáng
edges = cv2.Canny((obstacles*255).astype(np.uint8), 100, 200)
vis[edges > 0] = [255, 0, 255]

# Đường đi: Laser Xanh (Cyan)
vis[lbl > 0] = [255, 255, 0]

sy, sx = np.where(start_pt == 1)
gy, gx = np.where(goal_pt == 1)

if len(sy) > 0: cv2.circle(vis, (sx[0], sy[0]), 5, (0, 255, 0), -1)
if len(gy) > 0: cv2.circle(vis, (gx[0], gy[0]), 5, (0, 0, 255), -1)

vis_large = cv2.resize(vis, (w*3, h*3), interpolation=cv2.INTER_NEAREST)
cv2.imshow("Chaos World", vis_large)
cv2.waitKey(0)
cv2.destroyAllWindows()