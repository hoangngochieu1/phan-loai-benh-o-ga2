import streamlit as st
import torch
import numpy as np
import cv2
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
from ultralytics import YOLO
import timm
from data import disease_info

# Định nghĩa các biến đổi ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load mô hình YOLO và Xception
model_detect = YOLO('best.pt')
# model_classify = timm.create_model('xception', pretrained=False, num_classes=4)
model_classify = models.mobilenet_v2(pretrained=False)

model_classify.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(model_classify.last_channel, 4)
)
state_dict = torch.load('mobilenet.pth', map_location='cpu')

# Xử lý state_dict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    name = name.replace('fc.1', 'fc')
    new_state_dict[name] = v
model_classify.load_state_dict(new_state_dict, strict=False)


# Labels và thông tin bệnh
labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']


# Giao diện chính
st.set_page_config(page_title="Phát hiện bệnh qua phân gà",
                   page_icon="🐔", layout="wide")
st.sidebar.title("⚙️ Menu điều hướng")
st.sidebar.markdown("## 📋 Hướngg dẫn sử dụng")
st.sidebar.markdown("""
1. Tải lên ảnh phân gà hoặc chụp bằng camera.
2. Nhấn **Xử lý** để phát hiện vùng bệnh.
3. Xem chi tiết kết quả và cách phòng ngừa.
""")

st.title("🐔 Phát hiện bệnh qua phân gà")
st.markdown("### 🚀 **Ứng dụng AI hỗ trợ chẩn đoán bệnh gà nhanh chóng**")
st.markdown("---")

# **Tùy chọn tải ảnh**
option = st.radio("🖼️ Chọn cách tải ảnh:", options=[
                  "📤 Tải lên từ thiết bị", "📷 Chụp ảnh bằng camera"])
image = None

if option == "📤 Tải lên từ thiết bị":
    uploaded_file = st.file_uploader(
        "🌟 Tải lên ảnh (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
elif option == "Chụp ảnh bằng camera":
    camera_file = st.camera_input("📸 Chụp ảnh bằng camera")
    if camera_file:
        image = Image.open(camera_file).convert('RGB')

# **Xử lý khi có ảnh**
if image is not None:
    st.subheader("📂 Ảnh đầu vào")
    st.image(image, caption="📸 Ảnh đã chọn", use_column_width=True)

    if st.button("🔍 Xử lý ảnh", key="process_button"):
        try:
            # Dự đoán vùng phát hiện bệnh
            results = model_detect(image)
            if len(results[0].boxes) > 0:
                xmin, ymin, xmax, ymax = results[0].boxes.xyxy[0].cpu().numpy()
                img_crop = image.crop((xmin, ymin, xmax, ymax))

                # Chuẩn bị ảnh cho phân loại
                img_tensor = transform(img_crop).unsqueeze(0)
                model_classify.eval()
                with torch.no_grad():
                    predict = model_classify(img_tensor)
                    predicted_label = labels[torch.argmax(predict).item()]

                # Hiển thị kết quả
                st.subheader("📊 **Kết quả phân loại**")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_crop, caption="🔍 Khu vực phát hiện",
                             use_column_width=True)
                with col2:
                    st.success(f"🔬 **Loại bệnh phát hiện:** {predicted_label}")
                info = disease_info[predicted_label]
                st.markdown("---")
                st.write("### 🔎 **Thông tin chi tiết về bệnh**")
                st.header(info["name"])
                if 'description' in info:
                    st.write("### 📖 Mô tả:")
                    st.write(info["description"])
                if 'statistical' in info:
                    st.write("### 📊 Thống kê:")
                    st.write(info["statistical"])
                if 'causes' in info:
                    st.write("### 🧪 Nguyên nhân:")
                    st.write(info["causes"])
                if 'symptoms' in info:
                    st.write("### 🤒 Triệu chứng:")
                    st.write(info["symptoms"])
                if 'damage' in info:
                    st.write("### 💥 Tác hại:")
                    st.write(info["damage"])
                if 'prevention' in info:
                    st.write("### 🛡️ Cách phòng ngừa:")
                    st.write(info["prevention"])
                st.write("### 💊 Cách điều trị:")
                st.write(info["treatment"])
                # Vẽ khoanh vùng trên ảnh gốc
                image_np = np.array(image)
                image_np = cv2.rectangle(image_np, (int(xmin), int(ymin)),
                                         (int(xmax), int(ymax)), (0, 255, 0), 2)
                image_np = cv2.putText(image_np, predicted_label, (int(xmin), int(ymin) - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                st.image(image_np, caption="🖼️ Ảnh với vùng khoanh bệnh",
                         use_column_width=True)
            else:
                st.warning("Không phát hiện vùng bệnh nào trong ảnh.")
        except Exception as e:
            st.error(f"❌ Lỗi xử lý: {e}")
else:
    st.info("⚡ Vui lòng tải ảnh hoặc chụp ảnh để bắt đầu.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("#### 🌟 **Liên hệ hỗ trợ**")
st.sidebar.markdown(
    "[📧 Email: contact@hoangngochieutin92018@gmail.com](mailto:contact@hoangngochieutin92018@gmail.com)")
st.markdown("---")
st.markdown("🌟 **Cảm ơn bạn đã sử dụng ứng dụng của chúng tôi!**")