import numpy as np
import torch
import torch.nn as nn
import streamlit as st
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🔬")

CLASS_NAMES = {
    "akiec": "Актинический кератоз / болезнь Боуэна",
    "bcc": "Базальноклеточная карцинома",
    "bkl": "Доброкачественный кератоз",
    "df": "Дерматофиброма",
    "mel": "Меланома",
    "nv": "Меланоцитарный невус (родинка)",
    "vasc": "Сосудистое поражение",
}


@st.cache_resource
def load_model():
    checkpoint = torch.load("model.pth", map_location="cpu")
    classes = checkpoint["classes"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes


model, classes = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.title("🔬 Skin Lesion Classifier")
st.caption("Классификация дерматоскопических изображений по 7 категориям (HAM10000)")

st.warning(
    "⚠️ Учебный проект, не диагностический инструмент. "
    "Recall на меланоме ~75% — это значит, что каждый четвёртый случай меланомы "
    "модель может пропустить. Любое подозрительное образование — повод обратиться к дерматологу."
)

uploaded_file = st.file_uploader("Загрузи дерматоскопическое изображение", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", width=300)

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_idx = torch.argmax(probs).item()
    top_class = classes[top_idx]

    st.subheader(f"Наиболее вероятно: {CLASS_NAMES[top_class]}")
    st.metric("Уверенность модели", f"{probs[top_idx]:.0%}")

    st.subheader("Вероятности по всем классам")
    prob_dict = {CLASS_NAMES[c]: probs[i].item() for i, c in enumerate(classes)}
    prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
    st.bar_chart(prob_dict)
