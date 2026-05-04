import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

# ─────────────────────────────────────────────
# CLEAN CSS — remove all webrtc default junk
# ─────────────────────────────────────────────
st.markdown("""
<style>

/* Hide streamlit default header/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Body background */
body, .stApp {
    background-color: #0f0f0f;
    color: white;
}

/* Title styling */
h1 {
    text-align: center;
    font-size: 2rem;
    color: #ffffff;
    margin-bottom: 4px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #888888;
    font-size: 0.95rem;
    margin-bottom: 20px;
}

/* Hide ALL webrtc default buttons and controls */
.webrtc-container button {
    display: none !important;
}

/* The actual video box — make it clean */
video {
    border-radius: 12px !important;
    width: 100% !important;
    max-height: 480px !important;
    background: #1a1a1a !important;
    display: block !important;
}

/* Hide audio/video select dropdowns from webrtc */
.streamlit-webrtc-container select,
.streamlit-webrtc-container audio {
    display: none !important;
}

/* Start/Stop button styling */
div[data-testid="stButton"] > button {
    width: 100%;
    background-color: #1a1a1a;
    color: white;
    border: 2px solid #444;
    border-radius: 8px;
    padding: 10px;
    font-size: 1rem;
    font-weight: bold;
    cursor: pointer;
    margin-top: 10px;
    transition: all 0.2s;
}
div[data-testid="stButton"] > button:hover {
    background-color: #2a2a2a;
    border-color: #888;
}

/* Status badge */
.status-box {
    text-align: center;
    padding: 12px 20px;
    border-radius: 10px;
    font-size: 1.6rem;
    font-weight: bold;
    margin: 10px auto;
    max-width: 400px;
}
.mask-on {
    background-color: #0a2e0a;
    color: #00e676;
    border: 2px solid #00e676;
}
.mask-off {
    background-color: #2e0a0a;
    color: #ff5252;
    border: 2px solid #ff5252;
}

/* Remove webrtc component border/padding */
iframe {
    border: none !important;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_mask_model():
    return load_model("MobileNetV2_mask_model.h5")

model = load_mask_model()

# ─────────────────────────────────────────────
# SHARED PREDICTION STATE
# ─────────────────────────────────────────────
if "last_label" not in st.session_state:
    st.session_state.last_label = None
if "last_conf" not in st.session_state:
    st.session_state.last_conf = None


# ─────────────────────────────────────────────
# VIDEO PROCESSOR
# ─────────────────────────────────────────────
class MaskDetector(VideoProcessorBase):
    def __init__(self):
        self.label = "..."
        self.confidence = 0.0
        self.color_bgr = (200, 200, 200)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Preprocess
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        expanded = np.expand_dims(resized, axis=0)
        processed = preprocess_input(expanded.astype(np.float32))

        # Predict
        pred = model.predict(processed, verbose=0)[0][0]

        if pred < 0.5:
            self.label = "MASK"
            self.confidence = (1 - pred) * 100
            self.color_bgr = (0, 230, 100)   # green
        else:
            self.label = "NO MASK"
            self.confidence = pred * 100
            self.color_bgr = (50, 50, 255)   # red

        # ── Draw clean label on video ──────────────────────
        h, w = img.shape[:2]

        # Semi-transparent top bar
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        # Label text
        text = f"{self.label}  {self.confidence:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 48

        cv2.putText(img, text, (text_x, text_y),
                    font, font_scale, self.color_bgr, thickness, cv2.LINE_AA)

        # Thin colored border on video frame
        border_thickness = 6
        cv2.rectangle(img, (0, 0), (w - 1, h - 1),
                      self.color_bgr, border_thickness)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown("<h1>😷 Face Mask Detection</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time webcam detection using MobileNetV2</p>',
            unsafe_allow_html=True)

# Divider
st.markdown("---")

# WebRTC streamer — clean config
RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

ctx = webrtc_streamer(
    key="mask-detector",
    video_processor_factory=MaskDetector,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    # Clean button labels
    translations={
        "start": "▶  START CAMERA",
        "stop": "⏹  STOP",
        "select_device": "",
    }
)

# ─────────────────────────────────────────────
# LIVE STATUS BADGE (below video)
# ─────────────────────────────────────────────
status_placeholder = st.empty()

if ctx.video_processor:
    proc = ctx.video_processor
    label = proc.label
    conf = proc.confidence

    if label == "MASK":
        status_placeholder.markdown(
            f'<div class="status-box mask-on">✅ &nbsp; MASK &nbsp; {conf:.1f}%</div>',
            unsafe_allow_html=True
        )
    elif label == "NO MASK":
        status_placeholder.markdown(
            f'<div class="status-box mask-off">❌ &nbsp; NO MASK &nbsp; {conf:.1f}%</div>',
            unsafe_allow_html=True
        )
else:
    status_placeholder.markdown(
        '<div class="status-box" style="background:#1a1a1a; border:2px solid #333; color:#555;">'
        '📷 &nbsp; Click START CAMERA to begin'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# Small info footer
st.markdown("""
<p style="text-align:center; color:#444; font-size:0.8rem;">
MobileNetV2 · Trained from scratch · IIT Kharagpur · 2026
</p>
""", unsafe_allow_html=True)
