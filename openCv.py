import torch
from torch import nn
import torchvision.models
from PIL import Image
from typing import List, Tuple
from torchvision import transforms
import os
# import cv2
import streamlit as st
import urllib.request

# MODEL_URL = "https://huggingface.co/Calin224/fruit_classifier/resolve/main/efficientnetb2_food360.pth"
# MODEL_PATH = "models/efficientnetb2_food360.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

class_names = ['Apple 10', 'Apple 11', 'Apple 12', 'Apple 13', 'Apple 14', 'Apple 17', 'Apple 18', 'Apple 19', 'Apple 5', 'Apple 7', 'Apple 8', 'Apple 9', 'Apple Core 1', 'Apple Red Yellow 2', 'Apple worm 1', 'apple_6', 'apple_braeburn_1', 'apple_crimson_snow_1', 'apple_golden_1', 'apple_golden_2', 'apple_golden_3', 'apple_granny_smith_1', 'apple_hit_1', 'apple_pink_lady_1', 'apple_red_1', 'apple_red_2', 'apple_red_3', 'apple_red_delicios_1', 'apple_red_yellow_1', 'apple_rotten_1', 'Banana 3', 'Beans 1', 'Blackberrie 1', 'Blackberrie 2', 'Blackberrie half rippen 1', 'Blackberrie not rippen 1', 'Cabbage red 1', 'cabbage_white_1', 'Cactus fruit green 1', 'Cactus fruit red 1', 'Caju seed 1', 'carrot_1', 'Cherimoya 1', 'Cherry 3', 'Cherry 4', 'Cherry 5', 'Cherry Rainier 2', 'Cherry Rainier 3', 'Cherry Sour 1', 'Cherry Wax not ripen 1', 'Cherry Wax not ripen 2', 'Cherry Wax Red 2', 'Cherry Wax Red 3', 'Cucumber 10', 'Cucumber 9', 'cucumber_1', 'cucumber_3', 'eggplant_long_1', 'Gooseberry 1', 'pear_1', 'pear_3', 'Pistachio 1', 'Quince 2', 'Quince 3', 'Quince 4', 'Tomato 1', 'Tomato 10', 'Tomato 5', 'Tomato 7', 'Tomato 8', 'Tomato 9', 'Tomato Cherry Maroon 1', 'Tomato Cherry Orange 1', 'Tomato Cherry Red 2', 'Tomato Cherry Yellow 1', 'Tomato Maroon 2', 'zucchini_1', 'zucchini_dark_1']

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights)

model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
).to(device)

# if not os.path.exists(MODEL_PATH):
#     os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
#     with st.spinner("Downloading model from HuggingFace..."):
#         urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
#         st.success("Model downloaded successfully!")

model.load_state_dict(torch.load("models/efficientnetb2_food360.pth", map_location=device), strict=False)

model.to(device)
model.eval()

def pred_image(model: torch.nn.Module,
               img: Image.Image,
               class_names: List[str],
               image_size: Tuple[int, int] = (260, 260),
               transform: transforms.Compose = None,
               device: torch.device = device):
    if transform is not None:
        img_transform = transform
    else:
        img_transform = transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    model.to(device)
    model.eval()

    with torch.inference_mode():
        transformed_img = img_transform(img).unsqueeze(0).to(device)
        pred = model(transformed_img)
        probs = torch.softmax(pred, dim=1)
        pred_class = torch.argmax(probs, dim=1)

    return class_names[pred_class.item()]


st.title("Fruit Classification")
st.write("Upload an image of a fruit and see what it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image.", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Classifying..."):
            pred_class = pred_image(model=model.to(device),
                                    img=img,
                                    class_names=class_names,
                                    image_size=(260, 260),
                                    device=device)
        st.success(f"Predicted Class: {pred_class}")
