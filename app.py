
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

st.set_page_config(
    page_title="FairFace Age Group Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 FairFace CNN Age Group Classification")

st.markdown("""
This application demonstrates a CNN-based age group classification system trained using the FairFace dataset.

### Features
- Upload a face image
- Predict top 3 age-group classes
- Display confidence scores

### Responsible Use
This is an academic prototype only.
""")

age_names = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "more than 70"
]

device = torch.device("cpu")

class SimpleCNN(nn.Module):

    def __init__(self, num_classes=9):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 16 * 16, 256),

            nn.ReLU(),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


@st.cache_resource
def load_model():

    model = SimpleCNN(num_classes=9)

    checkpoint = torch.load(
        "balanced_sampling_model.pt",
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


model = load_model()

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

transform = transforms.Compose([

    transforms.Resize((128,128)),

    transforms.ToTensor(),

    transforms.Normalize(mean, std),
])

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        top_probs, top_classes = torch.topk(
            probabilities,
            3
        )

    st.subheader("Top 3 Predictions")

    for i in range(3):

        label = age_names[top_classes[0][i].item()]

        confidence = top_probs[0][i].item() * 100

        st.write(
            f"{i+1}. {label} — {confidence:.2f}%"
        )

st.markdown("---")

st.markdown("""
### Technical Details
- Model: Custom CNN
- Dataset: FairFace
- Accuracy: ~50%
- Framework: PyTorch + Streamlit

### Limitations
- Performance varies across demographic groups
- This is not a production-grade system
""")
