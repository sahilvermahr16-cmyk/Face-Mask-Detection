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

        # 1. Preprocessing
        processed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed = cv2.resize(processed, (224, 224))
        processed = np.expand_dims(processed, axis=0)
        processed = preprocess_input(processed)

        # 2. Prediction
        pred = model.predict(processed, verbose=0)[0][0]

        # 3. Label and Confidence Logic
        if pred < 0.5:
            label = "MASK 😷"
            # Confidence is how close it is to 0
            confidence = (1 - pred) * 100
            color = (0, 255, 0) # Green
        else:
            label = "NO MASK ❌"
            # Confidence is how close it is to 1
            confidence = pred * 100
            color = (0, 0, 255) # Red

        # 4. Text display on screen (Showing Percentage instead of raw float)
        display_text = f"{label} {confidence:.2f}%"
        cv2.putText(img, display_text, (20, 50),
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
