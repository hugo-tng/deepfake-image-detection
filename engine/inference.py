import os
from PIL import Image
import torch

from utils.config import TrainingConfig
from data.datasets import get_transforms
from data.facecrop import FaceCropper
import matplotlib.pyplot as plt

CLASS_NAMES = {
    0: 'Real',
    1: 'AI'
}

def preprocess_image(image_path, img_size: 224, device, cropper: FaceCropper = None):
    """
    Đọc và tiền xử lý ảnh giống hệt như lúc training (Resize -> Normalize).
    """
    if cropper is not None:
        # Cropper tự handle việc đọc file và trả về PIL Image
        pil_image = cropper(image_path)
        status_msg = "Processed Input (Face Crop)"
    else:
        # Fallback nếu không dùng crop
        pil_image = Image.open(image_path).convert('RGB')
        status_msg = "Original Input (No Crop)"

    plt.figure(figsize=(4, 4))
    plt.imshow(pil_image)
    plt.axis('off')
    plt.title(status_msg, fontsize=10)
    plt.show()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f" Không tìm thấy ảnh tại: {image_path}")

    # 2. Lấy bộ transform - test
    transforms_dict = get_transforms(img_size)
    transform = transforms_dict['test']

    # 3. Biến đổi ảnh thành tensor
    img_tensor = transform(pil_image)

    # 4. Thêm chiều batch (C, H, W) -> (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor.to(device)

def predict_image(models_list, image_path):
    """
    Hàm suy luận (Inference) cho ảnh đầu vào.
    
    Args:
        models: Danh sách mô hình 
        image_path: Đường dẫn ảnh
    """
    global_config = models_list[0]['config']    
    face_cropper = FaceCropper(
        out_size=256,         # Crop ra ảnh 256x256 cho nét
        target_face_ratio=1.2 # Zoom cận mặt (logic cũ)
    )

    # 1. Tiền xử lý
    print(f"\n--- Processing: {os.path.basename(image_path)} ---")
    img_tensor = preprocess_image(image_path, global_config.IMG_SIZE, global_config.DEVICE, cropper=face_cropper)

    results_data = []

    for entry in models_list:
        model = entry['model']
        model_name = entry['name']
        result = model_predict(model, img_tensor, model_name)
        results_data.append(result)

    # 3. In kết quả
    print(f"{'MODEL NAME':<25} | {'PREDICTION':<12} | {'CONFIDENCE':<12} | {'FAKE PROB':<12}")
    print("-" * 75)

    for row in results_data:
        print(f"{row['Model Name']:<25} | {row['Prediction']:<12} | {row['Confidence']:<12} | {row['Fake Prob']:<12}")
    
    print("-" * 75)

    return results_data


def model_predict(model, image_tensor, model_name):
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        outputs = model(image_tensor)
        
        # Tính xác suất (Softmax)
        probs = torch.softmax(outputs, dim=1)

        # Lấy class có xác suất cao nhất
        confidence, pred_class_idx = torch.max(probs, 1)

        conf_score = confidence.item()
        label = CLASS_NAMES.get(pred_class_idx.item(), "Unknown")
    
        # Lấy xác suất cụ thể của lớp AI (index 1)
        fake_prob = probs[0][1].item()

        result = {
            "Model Name": model_name,
            "Prediction": label.upper(),
            "Confidence": f"{conf_score*100:.2f}%",
            "Fake Prob": f"{fake_prob*100:.2f}%"
        }

    return result