import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Custom Styling
st.markdown(
    """
    <style>
    .stApp {
        background: url('https://static.vecteezy.com/system/resources/thumbnails/002/019/515/small_2x/house-rotating-background-free-video.jpg') no-repeat center center fixed;
        background-size: cover;
    }
    
    .stApp h1 {
        background-color: rgba(0, 0, 128, 0.7); 
        color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        font-size: 2.5em;
        text-align: center;
    }

    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.8);
        color: #000000;
        font-size: 1.2em;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.2em;
        border-radius: 10px;
        padding: 10px 24px;
        border: none;
    }

    .stButton {
        display: flex;
        justify-content: center;
    }

    .output-container {
        background-color: lightpink;
        color: black;
        font-size: 1.5em;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("House Price Prediction")

# Define ANN Model
class ANN_Model(nn.Module):
    def __init__(self, input_cols=11, hidden0=128, hidden1=128, hidden2=128,
                 hidden3=64, hidden4=64, hidden5=32, hidden6=16, output=1):
        super().__init__()
        self.f_connected0 = nn.Linear(input_cols, hidden0)
        self.f_connected1 = nn.Linear(hidden0, hidden1)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.f_connected3 = nn.Linear(hidden2, hidden3)
        self.f_connected4 = nn.Linear(hidden3, hidden4)
        self.f_connected5 = nn.Linear(hidden4, hidden5)
        self.f_connected6 = nn.Linear(hidden5, hidden6)
        self.out = nn.Linear(hidden6, output)

    def forward(self, x):
        x = F.relu(self.f_connected0(x))
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = F.relu(self.f_connected3(x))
        x = F.relu(self.f_connected4(x))
        x = F.relu(self.f_connected5(x))
        x = F.relu(self.f_connected6(x))
        x = self.out(x)
        return x

# Load the trained model
model = ANN_Model()
try:
    model.load_state_dict(torch.load("ANN_model.pth", map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    st.error(f"Failed to load the model: {e}")

# Input UI in 3 Columns
col1, col2, col3 = st.columns(3)

with col1:
    year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
    num_bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
    garage_type = st.selectbox("Garage Type", ['attached', 'detached', 'none'])
    garage_type_encoded = 2 if garage_type == 'attached' else 1 if garage_type == 'detached' else 0
    has_fireplace = st.selectbox("Has Fireplace", ["True", "False"])
    has_fireplace = 1 if has_fireplace == "True" else 0

with col2:
    num_of_bathrooms = st.number_input("Number of Bathrooms", min_value=0, max_value=10, value=2)
    total_sqft = st.number_input("Total Square Feet", min_value=100, max_value=10000, value=1800)
    garage_sqft = st.number_input("Garage Square Feet", min_value=0, max_value=2000, value=400)
    has_pool = st.selectbox("Has Pool", ["True", "False"])
    has_pool = 1 if has_pool == "True" else 0

with col3:
    has_central_heating = st.selectbox("Has Central Heating", ["True", "False"])
    has_central_heating = 1 if has_central_heating == "True" else 0
    has_central_cooling = st.selectbox("Has Central Cooling", ["True", "False"])
    has_central_cooling = 1 if has_central_cooling == "True" else 0
    zip_code = st.number_input("ZIP Code", min_value=10000, max_value=99999, value=11203)

# Prediction Logic
if st.button('Predict House Price'):
    input_features = np.array([
        year_built, num_bedrooms, total_sqft, garage_type_encoded, garage_sqft,
        has_fireplace, has_pool, has_central_heating, has_central_cooling, zip_code, num_of_bathrooms
    ], dtype=np.float32)

    input_tensor = torch.tensor(input_features).unsqueeze(0)  # Batch dimension

    with torch.no_grad():
        predicted_price = model(input_tensor).item()

    st.markdown(
        f'<div class="output-container">Estimated House Price: <b>${predicted_price:,.2f}</b></div>',
        unsafe_allow_html=True
    )
