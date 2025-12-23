import numpy as np
import cv2
import os
import random

# Lấy ngẫu nhiên
files = os.listdir("./dataset_urban_pro/inputs")
if not files: 
    print("Chưa có data!")
    exit()

IDX = random.choice(files).split('.')[0]

inp = np.load(f"./dataset_urban_pro/inputs/7.npy")
lbl = np.load(f"./dataset_urban_pro/labels/7.npy")

obstacles = inp[:, :, 0]
start_pt = inp[:, :, 1]
goal_pt = inp[:, :, 2]

h, w = obstacles.shape
vis = np.zeros((h, w, 3), dtype=np.uint8)

# Mặc định nền trắng xám (Sàn nhà)
vis[:] = [240, 240, 240] 

# Vật cản màu Xám Đậm có viền đen (giống bản vẽ kỹ thuật)
vis[obstacles == 1] = [80, 80, 80]
edges = cv2.Canny((obstacles*255).astype(np.uint8), 100, 200)
vis[edges > 0] = [0, 0, 0]

# Đường đi màu Xanh Dương
path_mask = (lbl > 0).astype(np.uint8)
vis[path_mask == 1] = [255, 100, 0] # Blue (BGR)

# Start / Goal
sy, sx = np.where(start_pt == 1)
gy, gx = np.where(goal_pt == 1)

if len(sy) > 0:
    cv2.circle(vis, (sx[0], sy[0]), 6, (0, 200, 0), -1) # Green
    cv2.putText(vis, "S", (sx[0]-4, sy[0]+4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

if len(gy) > 0:
    cv2.circle(vis, (gx[0], gy[0]), 6, (0, 0, 255), -1) # Red
    cv2.putText(vis, "G", (gx[0]-4, gy[0]+4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# Phóng to
vis_large = cv2.resize(vis, (w*3, h*3), interpolation=cv2.INTER_NEAREST)
cv2.imshow("Urban Pro Review", vis_large)
cv2.waitKey(0)
cv2.destroyAllWindows()