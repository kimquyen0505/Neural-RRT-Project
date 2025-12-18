import numpy as np
import cv2
import os

# Đường dẫn đến file muốn kiểm tra (ví dụ file số 0)
INPUT_FILE = "./dataset_complex/inputs/7.npy"
LABEL_FILE = "./dataset_complex/labels/7.npy"

def view_sample():
    # 1. Kiểm tra file có tồn tại không
    if not os.path.exists(INPUT_FILE):
        print(f"Lỗi: Không tìm thấy file {INPUT_FILE}")
        return

    # 2. Load dữ liệu từ file .npy
    inp = np.load(INPUT_FILE)   # Shape: (128, 128, 3)
    lbl = np.load(LABEL_FILE)   # Shape: (128, 128)

    print(f"Input Shape: {inp.shape}")
    print(f"Label Shape: {lbl.shape}")

    # 3. Tách các kênh thông tin
    # Input channel 0: Vật cản (Obstacles)
    # Input channel 1: Start point
    # Input channel 2: Goal point
    obstacles = inp[:, :, 0]
    start_pt = inp[:, :, 1]
    goal_pt = inp[:, :, 2]

    # 4. Tạo ảnh màu để hiển thị (Background đen)
    # OpenCV dùng format BGR (Blue-Green-Red)
    vis_img = np.zeros((128, 128, 3), dtype=np.uint8)

    # Tô màu Vật cản (Màu xám)
    vis_img[obstacles == 1] = [100, 100, 100]

    # Tô màu Đường đi (Màu Xanh Lá - Green)
    # Lưu ý: Label là đường đi A*
    vis_img[lbl == 1] = [0, 255, 0]

    # Tô màu Start (Màu Xanh Dương - Blue)
    # Dùng hàm np.where để tìm tọa độ điểm start
    sy, sx = np.where(start_pt == 1)
    if len(sy) > 0:
        cv2.circle(vis_img, (sx[0], sy[0]), 3, (255, 0, 0), -1)

    # Tô màu Goal (Màu Đỏ - Red)
    gy, gx = np.where(goal_pt == 1)
    if len(gy) > 0:
        cv2.circle(vis_img, (gx[0], gy[0]), 3, (0, 0, 255), -1)

    # 5. Phóng to ảnh lên 4 lần để dễ nhìn (128px hơi bé)
    vis_img_large = cv2.resize(vis_img, (512, 512), interpolation=cv2.INTER_NEAREST)

    # 6. Hiển thị
    print("Đang hiển thị... Nhấn phím bất kỳ vào cửa sổ ảnh để thoát.")
    cv2.imshow("Check Data (Green=Path, Grey=Wall)", vis_img_large)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    view_sample()