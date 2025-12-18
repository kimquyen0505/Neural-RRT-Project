import os
import shutil
import glob
from tqdm import tqdm # Nhớ pip install tqdm nếu chưa có

# --- CẤU HÌNH ---
TARGET_DIR = "./Final_Dataset" # Tên folder đích

def merge_all_datasets():
    # 1. Tìm tất cả các folder bắt đầu bằng "dataset" (trừ folder đích)
    # Cách này giúp tự động tìm dataset_ham_nguc, dataset_me_cung... mà không cần khai báo
    all_dirs = [d for d in os.listdir(".") if os.path.isdir(d) and d.startswith("dataset")]
    
    # Loại bỏ folder đích khỏi danh sách (để tránh đệ quy vô hạn)
    if "Final_Dataset" in all_dirs:
        all_dirs.remove("Final_Dataset")
        
    if not all_dirs:
        print("❌ Không tìm thấy folder dataset nào bắt đầu bằng 'dataset...'")
        return

    print(f"🔍 Tìm thấy {len(all_dirs)} nguồn dữ liệu: {all_dirs}")
    print("-" * 50)

    # 2. Tạo folder đích
    if os.path.exists(TARGET_DIR):
        # Hỏi user có muốn xóa cũ không để tránh gộp chồng chéo
        ans = input(f"⚠️ Folder '{TARGET_DIR}' đã tồn tại. Xóa đi làm lại từ đầu? (y/n): ")
        if ans.lower() == 'y':
            shutil.rmtree(TARGET_DIR)
        else:
            print("Đã hủy. Hãy backup folder cũ hoặc xóa tay.")
            return

    os.makedirs(f"{TARGET_DIR}/inputs")
    os.makedirs(f"{TARGET_DIR}/labels")

    # 3. Bắt đầu gộp
    global_counter = 0
    
    # Duyệt từng folder nguồn
    for folder in all_dirs:
        input_path = f"{folder}/inputs"
        label_path = f"{folder}/labels"
        
        # Kiểm tra cấu trúc folder có đúng không
        if not os.path.exists(input_path) or not os.path.exists(label_path):
            print(f"⚠️ Bỏ qua '{folder}' (Cấu trúc không hợp lệ, thiếu inputs/labels)")
            continue

        # Lấy danh sách file .npy
        files = glob.glob(f"{input_path}/*.npy")
        
        print(f"📂 Đang xử lý: {folder} ({len(files)} mẫu)...")
        
        # Copy từng file
        for f_path in tqdm(files):
            filename = os.path.basename(f_path)
            
            # Đường dẫn nguồn
            src_input = f"{input_path}/{filename}"
            src_label = f"{label_path}/{filename}"
            
            # Kiểm tra label tương ứng có tồn tại không
            if not os.path.exists(src_label):
                continue 

            # Đường dẫn đích (Đổi tên theo số thứ tự tăng dần)
            # Định dạng tên file 00001.npy để dễ sort
            dst_input = f"{TARGET_DIR}/inputs/{global_counter:06d}.npy"
            dst_label = f"{TARGET_DIR}/labels/{global_counter:06d}.npy"
            
            shutil.copyfile(src_input, dst_input)
            shutil.copyfile(src_label, dst_label)
            
            global_counter += 1

    # 4. Tạo file thông tin (metadata)
    with open(f"{TARGET_DIR}/info.txt", "w") as f:
        f.write(f"Total Samples: {global_counter}\n")
        f.write("Merged from:\n")
        for folder in all_dirs:
            f.write(f"- {folder}\n")

    print("=" * 50)
    print(f"✅ HOÀN TẤT! Tổng cộng: {global_counter} mẫu dữ liệu.")
    print(f"📍 Dữ liệu nằm tại: {TARGET_DIR}")
    print("Bạn đã sẵn sàng để Train AI!")

if __name__ == "__main__":
    merge_all_datasets()