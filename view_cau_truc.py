import numpy as np
import cv2
import os
import random

files = os.listdir("./dataset_cau_truc/inputs")
if not files: exit()

IDX = random.choice(files).split('.')[0]

inp = np.load(f"./dataset_cau_truc/inputs/8.npy")
lbl = np.load(f"./dataset_cau_truc/labels/8.npy")

obstacles = inp[:, :, 0]
start_pt = inp[:, :, 1]
goal_pt = inp[:, :, 2]

h, w = obstacles.shape
vis = np.zeros((h, w, 3), dtype=np.uint8)

# Chế độ màu sắc tương phản cao (High Contrast)
vis[:] = [255, 255, 255] # Nền trắng

# Vật cản đen tuyền
vis[obstacles == 1] = [0, 0, 0]

# Đường đi: Heatmap màu
path_mask = (lbl > 0).astype(np.uint8)
vis[path_mask == 1] = [255, 0, 100] # Deep Pink

sy, sx = np.where(start_pt == 1)
gy, gx = np.where(goal_pt == 1)

if len(sy) > 0: cv2.circle(vis, (sx[0], sy[0]), 6, (0, 200, 0), -1)
if len(gy) > 0: cv2.circle(vis, (gx[0], gy[0]), 6, (0, 0, 255), -1)

vis_large = cv2.resize(vis, (w*3, h*3))
cv2.imshow("Exotic Map View", vis_large)
cv2.waitKey(0)
cv2.destroyAllWindows()