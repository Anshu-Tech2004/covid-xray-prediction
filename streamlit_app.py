# streamlit_app.py

import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io

# Load the model
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 1)
model.load_state_dict(torch.load('covid_model.pth', weights_only=True))
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_covid(image):
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = torch.sigmoid(model(image))
        prediction = (output > 0.5).float()
    return prediction.item()

st.title("COVID-19 X-ray Prediction")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    if st.button("Predict"):
        prediction = predict_covid(image)
        if prediction == 1:
            prediction_text = "<h2 style='color: green;'>Prediction: COVID-19 Negative</h2>"
        else:
            prediction_text = "<h2 style='color: red;'>Prediction: COVID-19 Positive</h2>"
        st.markdown(prediction_text, unsafe_allow_html=True)

# Add an expander for information
with st.expander("About this application"):
    st.write("This application uses a deep learning model(ResNet) to predict COVID-19 from X-ray images.")
    st.write("The model may not be perfectly accurate, and should not be used as a replacement for medical professionals.")

# Add some CSS styling
st.markdown(
    """
    <style>
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)