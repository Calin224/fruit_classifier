import torch
from torch import nn
import torchvision.models
from PIL import Image
from typing import List, Tuple
from torchvision import transforms
import os
# import cv2
import streamlit as st

device = "cuda" if torch.cuda.is_available() else "cpu"

weights = torchvision.models.ResNet18_Weights.DEFAULT
model = torchvision.models.resnet18(weights=weights)

model.fc = nn.Sequential(
    nn.Linear(in_features=512, out_features=194)
)

model.load_state_dict(torch.load("models/resnet18_100x100.pth", map_location=device))

model.to(device)
model.eval()

class_names_local = sorted(next(os.walk("data/Training"))[1])

class_names = [x for x in class_names_local]

# transform = transforms.Compose([
#     transforms.Resize((100, 100)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_YCrCb2RGB))
#     img_tensor = transform(img).unsqueeze(0).to(device)
#
#     with torch.inference_mode():
#         pred = model(img_tensor)
#         probs = torch.softmax(pred, dim=1)
#         pred_class = torch.argmax(probs, dim=1)
#         label = class_names[pred_class]
#         confidence = probs[0][pred_class].item()
#
#     cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#     cv2.imshow("Live Fruit Detection", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

def pred_image(model: torch.nn.Module,
               img: Image.Image,
               class_names: List[str],
               image_size: Tuple[int, int] = (100, 100),
               transform: transforms.Compose = None,
               device: torch.device = device):

    if transform is not None:
        img_transform = transform
    else:
        img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    model.to(device)
    model.eval()

    with torch.inference_mode():
        tranformed_img = img_transform(img).unsqueeze(0).to(device)
        pred = model(tranformed_img)
        probs = torch.softmax(pred, dim=1)
        pred_class = torch.argmax(probs, dim=1)

    return class_names[pred_class.item()]

st.title("Fruit Classification")
st.write("Upload an image of a fruit and see what it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image.", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Classifying..."):
            pred_class = pred_image(model=model,
                                    img=img,
                                    class_names=class_names,
                                    image_size=(100, 100),
                                    device=device)
        st.success(f"Predicted Class: {pred_class}")