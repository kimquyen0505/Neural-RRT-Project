import numpy as np
import cv2
import os
import random

files = os.listdir("./dataset_special/inputs")
if not files: exit()

IDX = random.choice(files).split('.')[0]

inp = np.load(f"./dataset_special/inputs/5.npy")
lbl = np.load(f"./dataset_special/labels/5.npy")

obstacles = inp[:, :, 0]
start_pt = inp[:, :, 1]
goal_pt = inp[:, :, 2]

h, w = obstacles.shape
vis = np.zeros((h, w, 3), dtype=np.uint8)

# Chế độ xem: Heatmap hồng ngoại (Thermal Style)
# Nền lạnh (Xanh tím)
vis[:] = [50, 0, 50]

# Vật cản nóng (Đỏ cam)
vis[obstacles == 1] = [0, 165, 255] # Orange

# Đường đi lạnh nhất (Trắng sáng)
path_mask = (lbl > 0).astype(np.uint8)
vis[path_mask == 1] = [255, 255, 255]

sy, sx = np.where(start_pt == 1)
gy, gx = np.where(goal_pt == 1)

if len(sy) > 0: cv2.circle(vis, (sx[0], sy[0]), 6, (0, 255, 0), -1)
if len(gy) > 0: cv2.circle(vis, (gx[0], gy[0]), 6, (0, 255, 255), -1)

vis_large = cv2.resize(vis, (w*3, h*3))
cv2.imshow("Special Map Thermal View", vis_large)
cv2.waitKey(0)
cv2.destroyAllWindows()