import numpy as np
import cv2
import os
import random

files = os.listdir("./dataset_hybrid/inputs")
if not files: exit()

IDX = random.choice(files).split('.')[0]

inp = np.load(f"./dataset_hybrid/inputs/4.npy")
lbl = np.load(f"./dataset_hybrid/labels/4.npy")

obstacles = inp[:, :, 0]
start_pt = inp[:, :, 1]
goal_pt = inp[:, :, 2]

h, w = obstacles.shape
vis = np.zeros((h, w, 3), dtype=np.uint8)

# Nền: Gradient hoặc màu tối
vis[:] = [40, 40, 45]

# Vẽ tường
vis[obstacles == 1] = [150, 150, 180] # Tường xám sáng

# Hiệu ứng phân vùng (Visual Trick)
# Vẽ đường kẻ mờ chia map để dễ nhìn
mid = w // 2
cv2.line(vis, (mid, 0), (mid, h), (60, 60, 70), 1)
cv2.line(vis, (0, mid), (w, mid), (60, 60, 70), 1)

# Đường đi: Màu Cầu Vồng (Rainbow Path) để nổi bật
path_mask = (lbl > 0).astype(np.uint8)
vis[path_mask == 1] = [0, 255, 128] # Spring Green

# Start / Goal
sy, sx = np.where(start_pt == 1)
gy, gx = np.where(goal_pt == 1)
if len(sy) > 0: 
    cv2.circle(vis, (sx[0], sy[0]), 6, (0, 0, 255), -1) # Start Đỏ
    cv2.putText(vis, "START", (sx[0], sy[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

if len(gy) > 0: 
    cv2.circle(vis, (gx[0], gy[0]), 6, (0, 255, 255), -1) # Goal Vàng
    cv2.putText(vis, "GOAL", (gx[0], gy[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

vis_large = cv2.resize(vis, (w*3, h*3))
cv2.imshow("Hybrid Visualization", vis_large)
cv2.waitKey(0)
cv2.destroyAllWindows()