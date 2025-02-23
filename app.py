import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from models import MLP, CNN, LeNet5

#load models
mlp_model = MLP()
cnn_model = CNN()
lenet_model = LeNet5()

mlp_model.load_state_dict(torch.load('mlp_model_weights.pth'))
cnn_model.load_state_dict(torch.load('cnn_model_weights.pth'))
lenet_model.load_state_dict(torch.load('lenet_model_weights.pth'))

mlp_model.eval()
cnn_model.eval()
lenet_model.eval()

#transform image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict_digit(image, model):
    with torch.no_grad():
        image = transform(image).unsqueeze(0)
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()

#streamlit app layout
st.set_page_config(page_title="Handwritten Digit Recognition", layout="wide")
st.markdown("""
    <style>
        .title { text-align: center; font-size: 40px; color: #4CAF50; font-weight: bold; margin-top: 10px; }
        .description { text-align: center; font-size: 18px; color: white; margin-bottom: 30px; }
        .button { background-color: #4CAF50; color: white; padding: 12px 20px; border-radius: 5px; font-size: 16px; transition: background-color 0.3s ease; }
        .button:hover { background-color: #45a049; }
        .container { display: flex; justify-content: center; align-items: center; flex-direction: column; padding-top: 20px; }
        .left-container { width: 35%; padding: 20px; background-color: #f1f8e9; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-right: 30px; }
        .right-container { width: 60%; padding: 20px; background-color: #ffffff; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .sidebar { background-color: #4CAF50; color: white; padding: 20px; border-radius: 10px; }
        .sidebar h3 { color: #fff; }
        .section-title { font-size: 20px; font-weight: bold; margin-bottom: 10px; }
        .section-description { font-size: 16px; margin-bottom: 20px; }
        .instructions-title { color: #ff9800; }
        .instructions-description { color: #ffffff; }
        .upload-title { color: #2196f3; }
        .model-title { color: #9c27b0; }
        .center-button { display: flex; justify-content: center; width: 100%; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">Handwritten Digit Recognition</h1>', unsafe_allow_html=True)
with st.sidebar:
    st.markdown('<p class="section-title instructions-title">Instructions</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-description instructions-description">Please upload an image of a handwritten digit and select a model for prediction.</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title upload-title">Upload Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a digit image:", type=["jpg", "png", "jpeg"])
    st.markdown('<p class="section-title model-title">Model Selection</p>', unsafe_allow_html=True)
    model_option = st.selectbox("Choose a model for digit prediction:", ("MLP", "CNN", "LeNet5"))
container = st.container()

with container:
    st.markdown('<h3 style="text-align:center; color: #2196f3;">Uploaded Image</h3>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='', width=300)
        st.markdown('<div class="center-button">', unsafe_allow_html=True)
        if st.button("Predict", key="predict"):
            with st.spinner('Predicting...'):
                if model_option == "MLP":
                    result = predict_digit(image, mlp_model)
                elif model_option == "CNN":
                    result = predict_digit(image, cnn_model)
                elif model_option == "LeNet5":
                    result = predict_digit(image, lenet_model)
                st.success(f"The predicted digit is: {result}", icon="✅")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="text-align: center; color: #f44336; font-size: 18px;">Please upload an image to get started</p>', unsafe_allow_html=True)
