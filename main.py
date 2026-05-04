import streamlit as st
import cv2
import numpy as np
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
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

body, .stApp {
    background-color: #0f0f0f;
    color: white;
}

h1 {
    text-align: center;
    font-size: 2rem;
    color: #ffffff;
    margin-bottom: 4px;
}

.subtitle {
    text-align: center;
    color: #888888;
    font-size: 0.95rem;
    margin-bottom: 20px;
}

video {
    border-radius: 12px !important;
    width: 100% !important;
    background: #1a1a1a !important;
    display: block !important;
}

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
.waiting {
    background-color: #1a1a1a;
    color: #555555;
    border: 2px solid #333333;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL — using keras directly (no tf import)
# ─────────────────────────────────────────────
@st.cache_resource
def load_mask_model():
    # Import inside function to avoid top-level TF crash
    from tensorflow import keras
    model = keras.models.load_model("MobileNetV2_mask_model.h5")
    return model


model = load_mask_model()


# ─────────────────────────────────────────────
# VIDEO PROCESSOR
# ─────────────────────────────────────────────
class MaskDetector(VideoProcessorBase):

    def __init__(self):
        self.label = "..."
        self.confidence = 0.0
        self.color_bgr = (150, 150, 150)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # ── Preprocess ──────────────────────────────
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        arr = resized.astype(np.float32)

        # MobileNetV2 preprocess_input: scale to [-1, 1]
        arr = (arr / 127.5) - 1.0
        arr = np.expand_dims(arr, axis=0)

        # ── Predict ─────────────────────────────────
        pred = model.predict(arr, verbose=0)[0][0]

        if pred < 0.5:
            self.label = "MASK"
            self.confidence = (1 - pred) * 100
            self.color_bgr = (0, 230, 100)    # green
        else:
            self.label = "NO MASK"
            self.confidence = pred * 100
            self.color_bgr = (50, 50, 255)    # red

        # ── Draw on frame ───────────────────────────
        h, w = img.shape[:2]

        # Semi-transparent top bar
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 65), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        # Label text centered
        text = f"{self.label}  {self.confidence:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.1
        thick = 2
        tw, th = cv2.getTextSize(text, font, scale, thick)[0]
        tx = (w - tw) // 2
        cv2.putText(img, text, (tx, 44), font, scale,
                    self.color_bgr, thick, cv2.LINE_AA)

        # Colored border
        cv2.rectangle(img, (0, 0), (w - 1, h - 1),
                      self.color_bgr, 5)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown("<h1>😷 Face Mask Detection</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Real-time webcam · MobileNetV2 · IIT Kharagpur</p>',
    unsafe_allow_html=True
)
st.markdown("---")

RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

ctx = webrtc_streamer(
    key="mask-detector",
    video_processor_factory=MaskDetector,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    translations={
        "start": "▶  START CAMERA",
        "stop":  "⏹  STOP",
        "select_device": "",
    }
)

# ── Status badge below video ─────────────────
status = st.empty()

if ctx.video_processor:
    proc = ctx.video_processor
    if proc.label == "MASK":
        status.markdown(
            f'<div class="status-box mask-on">✅ &nbsp; MASK &nbsp; {proc.confidence:.1f}%</div>',
            unsafe_allow_html=True
        )
    elif proc.label == "NO MASK":
        status.markdown(
            f'<div class="status-box mask-off">❌ &nbsp; NO MASK &nbsp; {proc.confidence:.1f}%</div>',
            unsafe_allow_html=True
        )
    else:
        status.markdown(
            '<div class="status-box waiting">⏳ &nbsp; Detecting...</div>',
            unsafe_allow_html=True
        )
else:
    status.markdown(
        '<div class="status-box waiting">📷 &nbsp; Click START CAMERA to begin</div>',
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#333;font-size:0.8rem;">'
    'Sahil Verma · 24MA40023 · M.Sc. Mathematics · IIT Kharagpur · 2026'
    '</p>',
    unsafe_allow_html=True
)
