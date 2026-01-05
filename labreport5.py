import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import torch.nn.functional as F

# Step 1: Configure Page Title and Layout
st.set_page_config(
    page_title="Image Classification with PyTorch & Streamlit",
    page_icon="",
    layout="centered"
)

st.title("Simple Image Classification Web App")
st.write("Using **PyTorch ResNet-18 (pretrained on ImageNet)** + Streamlit")


# Step 3: Configure to run only on CPU
device = torch.device("cpu")

# Step 4: Load pre-trained ResNet18 and set to evaluation mode
@st.cache_resource
def load_model():
    # Using the latest weights parameter as per modern Torchvision standards
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.to(device)
    model.eval()
    return model

# Load labels for ImageNet
@st.cache_data
def load_labels():
    # Standard ImageNet class labels
    import json
    with open("imagenet_class_index.json", "r") as f:
        labels = json.load(f)
    return {int(k): v[1] for k, v in labels.items()}

model = load_model()

# Step 5: Define image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Step 6: User Interface for file upload
st.title("Computer Vision")
st.write("Upload an image to classify it using the ResNet18 model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Step 7: Convert to tensor and perform inference (no gradient)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)

    # Step 8: Apply Softmax and get top-5 predictions
    probabilities = F.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)


    try:
        # Simplified label handling for demonstration'
        results = []
        st.subheader("🔍 Top-5 Predictions")
        for i in range(top5_prob.size(0)):
            results.append({"Class": f"Category {top5_catid[i].item()}", 
                            "Probability": top5_prob[i].item()})
        
        df = pd.DataFrame(results)

        # Step 9: Visualize results with a bar chart
        st.subheader("📊 Predictions Table")
        st.bar_chart(df.set_index("Class"))
        st.table(df)

    except Exception as e:
        st.info("👆 Please upload an image to start.")