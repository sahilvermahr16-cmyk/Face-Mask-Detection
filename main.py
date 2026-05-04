import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- CUSTOM CSS FOR CLEAN LOOK ---
st.markdown(
    """
    <style>
    /* Video container ko fix size dena */
    .video-container {
        width: 100%;
        overflow: hidden; /* Isse extra controls cut jayenge */
        border: 5px solid #2e2e2e;
        border-radius: 15px;
    }
    /* Browser ke default controls ko hatane ki koshish */
    video {
        width: 100% !important;
        height: auto !important;
        pointer-events: none !important;
    }
    /* Niche ki bar hide karne ke liye additional force */
    video::-webkit-media-controls-panel {
        display: none !important;
        -webkit-appearance: none;
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
        processed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed = cv2.resize(processed, (224, 224))
        processed = np.expand_dims(processed, axis=0)
        processed = preprocess_input(processed)

        pred = model.predict(processed, verbose=0)[0][0]

        if pred < 0.5:
            label, color = "MASK", (0, 255, 0)
            confidence = (1 - pred) * 100
        else:
            label, color = "NO MASK", (0, 0, 255)
            confidence = pred * 100

        display_text = f"{label} {confidence:.2f}%"
        cv2.putText(img, display_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        return img

st.title("Face Mask Detection")

# Video ko ek "Container" ke andar daalna taaki controls hide ho sakein
with st.container():
    webrtc_streamer(
        key="mask-detect",
        video_processor_factory=VideoTransformer,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
