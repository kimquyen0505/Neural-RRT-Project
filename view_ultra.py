import numpy as np
import cv2
import os
import random

files = os.listdir("./dataset_ultra/inputs")
if not files: exit()

IDX = random.choice(files).split('.')[0]
print(f"Viewing Ultra Map: {IDX}")

inp = np.load(f"./dataset_ultra/inputs/{IDX}.npy")
lbl = np.load(f"./dataset_ultra/labels/{IDX}.npy")

obstacles = inp[:, :, 0]
start_pt = inp[:, :, 1]
goal_pt = inp[:, :, 2]

h, w = obstacles.shape
vis = np.zeros((h, w, 3), dtype=np.uint8)

# Chế độ màu sắc: Artistic
vis[:] = [20, 20, 20]

# Vật cản màu tím Neon
vis[obstacles == 1] = [255, 0, 200] 

# Đường đi màu xanh ngọc
path_mask = (lbl > 0).astype(np.uint8)
vis[path_mask == 1] = [0, 255, 200] 

sy, sx = np.where(start_pt == 1)
gy, gx = np.where(goal_pt == 1)

if len(sy) > 0: cv2.circle(vis, (sx[0], sy[0]), 6, (0, 0, 255), -1)
if len(gy) > 0: cv2.circle(vis, (gx[0], gy[0]), 6, (255, 255, 0), -1)

vis_large = cv2.resize(vis, (w*3, h*3))
cv2.imshow("Ultra Map Art", vis_large)
cv2.waitKey(0)
cv2.destroyAllWindows()