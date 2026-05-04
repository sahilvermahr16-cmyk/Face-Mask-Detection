import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- ULTIMATE CSS FIX (Isse sirf box dikhega, video controls nahi) ---
st.markdown(
    """
    <style>
    /* 1. Niche ki controls bar aur buttons ko hide karo */
    video::-webkit-media-controls {
        display: none !important;
    }
    video::-webkit-media-controls-enclosure {
        display: none !important;
    }
    video::-webkit-media-controls-panel {
        display: none !important;
    }
    
    /* 2. Video frame ko simple box ki tarah dikhao */
    video {
        pointer-events: none !important; 
        border: 2px solid #555;
        border-radius: 10px;
        width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_mask_model():
    return load_model("MobileNetV2_mask_model.h5")

model = load_mask_model()

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Preprocessing
        processed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed = cv2.resize(processed, (224, 224))
        processed = np.expand_dims(processed, axis=0)
        processed = preprocess_input(processed)

        # Prediction
        pred = model.predict(processed, verbose=0)[0][0]

        if pred < 0.5:
            label, color = "MASK", (0, 255, 0)
            confidence = (1 - pred) * 100
        else:
            label, color = "NO MASK", (0, 0, 255)
            confidence = pred * 100

        # Output label display
        display_text = f"{label} {confidence:.2f}%"
        cv2.putText(img, display_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3) # Thoda thick font
        
        return img

st.title("Face Mask Detection")
st.write("Live Camera Feed:")

# Streamer Setup
webrtc_streamer(
    key="mask-detect",
    video_processor_factory=VideoTransformer,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    # Audio False karna sabse important hai clean box ke liye
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
