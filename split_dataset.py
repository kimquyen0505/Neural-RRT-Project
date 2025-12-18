import os
import shutil
import random
import glob
from tqdm import tqdm

# --- CẤU HÌNH ---
SOURCE_DIR = "./Final_Dataset"       # Folder chứa dữ liệu gộp
TARGET_DIR = "./Dataset_Train" # Folder đầu ra
TRAIN_RATIO = 0.8                    # Tỷ lệ train (80%)

def split_dataset():
    # 1. Kiểm tra nguồn
    if not os.path.exists(SOURCE_DIR):
        print(f" Không tìm thấy folder nguồn: {SOURCE_DIR}")
        print("Hãy chạy merge_final_dataset.py trước!")
        return

    # 2. Chuẩn bị folder đích (Xóa cũ tạo mới)
    if os.path.exists(TARGET_DIR):
        print(f" Đang dọn dẹp folder cũ '{TARGET_DIR}'...")
        shutil.rmtree(TARGET_DIR)
    
    # Tạo cấu trúc thư mục chuẩn cho PyTorch/TensorFlow
    os.makedirs(f"{TARGET_DIR}/train/inputs")
    os.makedirs(f"{TARGET_DIR}/train/labels")
    os.makedirs(f"{TARGET_DIR}/val/inputs")
    os.makedirs(f"{TARGET_DIR}/val/labels")

    # 3. Lấy danh sách file và TRỘN NGẪU NHIÊN
    files = glob.glob(f"{SOURCE_DIR}/inputs/*.npy")
    # Chỉ lấy tên file (ví dụ: 0001.npy)
    filenames = [os.path.basename(f) for f in files]
    
    print(f"Đang trộn ngẫu nhiên {len(filenames)} mẫu dữ liệu...")
    random.shuffle(filenames) # Bước quan trọng nhất!

    # 4. Tính toán số lượng
    total_files = len(filenames)
    train_count = int(total_files * TRAIN_RATIO)
    
    train_files = filenames[:train_count]
    val_files = filenames[train_count:]
    
    print(f" Chia dữ liệu: Train ({len(train_files)}) - Val ({len(val_files)})")

    # 5. Hàm copy file
    def copy_files(file_list, split_type):
        print(f"Đang tạo tập {split_type.upper()}...")
        for name in tqdm(file_list):
            # Nguồn
            src_in = f"{SOURCE_DIR}/inputs/{name}"
            src_lbl = f"{SOURCE_DIR}/labels/{name}"
            
            # Đích
            dst_in = f"{TARGET_DIR}/{split_type}/inputs/{name}"
            dst_lbl = f"{TARGET_DIR}/{split_type}/labels/{name}"
            
            shutil.copyfile(src_in, dst_in)
            shutil.copyfile(src_lbl, dst_lbl)

    # Thực hiện copy
    copy_files(train_files, "train")
    copy_files(val_files, "val")

    print("="*50)
    print("ĐÃ XONG! Dữ liệu sẵn sàng tại:", TARGET_DIR)
    print("Cấu trúc thư mục:")
    print(f"   {TARGET_DIR}/")
    print("     ├── train/")
    print("     │     ├── inputs/ (Chứa map)")
    print("     │     └── labels/ (Chứa đường đi)")
    print("     └── val/")
    print("           ├── inputs/")
    print("           └── labels/")

if __name__ == "__main__":
    split_dataset()