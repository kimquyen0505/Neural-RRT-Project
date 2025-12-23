import torch
import os
import glob

print("=== KIỂM TRA GPU ===")
if torch.cuda.is_available():
    print(f"✅ Đã nhận diện GPU: {torch.cuda.get_device_name(0)}")
    try:
        # Test thử khả năng tính toán của GPU với kiến trúc mới
        x = torch.rand(5, 5).cuda()
        print("✅ Test tính toán Tensor trên GPU: OK")
    except Exception as e:
        print(f"❌ GPU có lỗi tương thích: {e}")
        print("👉 GIẢI PHÁP: Cài lại PyTorch Nightly/Preview (CUDA 12.x)")
else:
    print("❌ Không tìm thấy GPU! PyTorch đang chạy bằng CPU.")

print("\n=== KIỂM TRA DỮ LIỆU ===")
DATA_DIR = "./Dataset_Ready_For_Train"
abs_path = os.path.abspath(DATA_DIR)
print(f"Đang tìm dữ liệu tại: {abs_path}")

if not os.path.exists(DATA_DIR):
    print("❌ LỖI: Không tìm thấy folder 'Dataset_Ready_For_Train'!")
    print(f"👉 Hãy copy folder dữ liệu vào: {os.path.dirname(abs_path)}")
else:
    train_files = glob.glob(f"{DATA_DIR}/train/inputs/*.npy")
    val_files = glob.glob(f"{DATA_DIR}/val/inputs/*.npy")
    
    if len(train_files) > 0:
        print(f"✅ Tìm thấy {len(train_files)} file train.")
        print(f"✅ Tìm thấy {len(val_files)} file val.")
        print("=> Dữ liệu OK!")
    else:
        print("❌ LỖI: Folder tồn tại nhưng KHÔNG CÓ FILE .NPY bên trong!")
        print("👉 Hãy kiểm tra lại folder train/inputs")