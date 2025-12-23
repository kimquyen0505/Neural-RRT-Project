import numpy as np
import cv2
import os
import random

# Random xem 1 map bất kỳ trong folder
files = os.listdir("./dataset_me_cung/inputs")
if not files:
    print("Chưa có data!")
    exit()

IDX = random.choice(files).split('.')[0] # Lấy ID ngẫu nhiên

inp = np.load(f"./dataset_me_cung/inputs/7.npy")
lbl = np.load(f"./dataset_me_cung/labels/7.npy")

obstacles = inp[:, :, 0]
start_pt = inp[:, :, 1]
goal_pt = inp[:, :, 2]

# Vẽ
vis = np.zeros((256, 256, 3), dtype=np.uint8)
vis[:] = [20, 20, 20] # Nền đen xám

# Tường màu Cam Đất
vis[obstacles == 1] = [50, 100, 200] 

# Đường đi màu Xanh Neon
vis[lbl > 0] = [0, 255, 200]

# Start / Goal
sy, sx = np.where(start_pt == 1)
gy, gx = np.where(goal_pt == 1)
if len(sy) > 0: cv2.circle(vis, (sx[0], sy[0]), 5, (255, 255, 0), -1)
if len(gy) > 0: cv2.circle(vis, (gx[0], gy[0]), 5, (0, 0, 255), -1)

# Phóng to
vis_large = cv2.resize(vis, (512, 512), interpolation=cv2.INTER_NEAREST)
cv2.imshow("Random Maze Check", vis_large)
cv2.waitKey(0)
cv2.destroyAllWindows()