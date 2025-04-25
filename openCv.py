import torch
from torch import nn
import torchvision.models
from PIL import Image
from typing import List, Tuple
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

weights = torchvision.models.ResNet18_Weights.DEFAULT
model = torchvision.models.resnet18(weights=weights)

model.fc = nn.Sequential(
    nn.Linear(in_features=512, out_features=194)
)

model.load_state_dict(torch.load("models/resnet18_100x100.pth", map_location=device))

model.to(device)
model.eval()

# def pred_and_plot_image(model: torch.nn.Module,
#                         image_path: str,
#                         class_names: List[str],
#                         image_size: Tuple[int, int]=(100,100),
#                         transform: torchvision.transforms=None,
#                         device: torch.device=device):
#     img = Image.open(image_path)
#
#     if transform is not None:
#         image_transform = transform
#     else:
#         image_transform = transforms.Compose([
#             transforms.Resize(image_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])
#
#     model.to(device)
#
#     model.eval()
#     with torch.inference_mode():
#         transformed_img = image_transform(img).unsqueeze(0)
#
#         target_image_pred = model(transformed_img.to(device))
#
#         target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
#
#         target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
#
#     return class_names[target_image_pred_label.item()]


class_names = sorted(next(os.walk("data/Training"))[1])

# print(pred_and_plot_image(model=model,
#                     image_path="data/Test/Apple 8/r0_3_100.jpg",
#                     class_names=class_names,
#                     image_size=(100,100),
#                     device=device))

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_YCrCb2RGB))
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        pred = model(img_tensor)
        probs = torch.softmax(pred, dim=1)
        pred_class = torch.argmax(probs, dim=1)
        label = class_names[pred_class]
        confidence = probs[0][pred_class].item()

    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Fruit Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

