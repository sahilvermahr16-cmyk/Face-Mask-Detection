import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Load model with caching to prevent reloading
@st.cache_resource
def load_mask_model():
    return load_model("MobileNetV2_mask_model.h5")

model = load_mask_model()

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Preprocessing (Model input size 224x224)
        processed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed = cv2.resize(processed, (224, 224))
        processed = np.expand_dims(processed, axis=0)
        processed = preprocess_input(processed)

        # Prediction
        pred = model.predict(processed, verbose=0)[0][0]

        # Mapping: pred < 0.5 is Mask, else No Mask
        label, color = ("MASK 😷", (0, 255, 0)) if pred < 0.5 else ("NO MASK ❌", (0, 0, 255))

        # Text display on screen
        cv2.putText(img, f"{label} ({pred:.2f})", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return img

# 2. Streamlit UI
st.title("Face Mask Detection Web App")
st.write("Click 'Start' to begin webcam detection")

webrtc_streamer(
    key="mask-detect",
    video_processor_factory=VideoTransformer,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)