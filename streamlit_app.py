import streamlit as st
import av
import cv2
import numpy as np
import pickle
import mediapipe as mp
import threading
import time

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from tensorflow.keras.models import load_model

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Real-Time Sign Language Translator",
    layout="wide"
)

st.title("🤟 Real-Time Sign Language Translator")

# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_resources():

    model = load_model("models/sign_model.keras")

    with open("models/label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, encoder

model, encoder = load_resources()

# =====================================
# MEDIAPIPE
# =====================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# =====================================
# SHARED STATE
# =====================================

class SharedState:

    def __init__(self):

        self.lock = threading.Lock()

        self.current_sign = ""
        self.confidence = 0

        self.sentence = []

        self.last_sign = ""
        self.last_added_time = 0

shared_state = SharedState()

# =====================================
# VIDEO PROCESSOR
# =====================================

class SignProcessor(VideoProcessorBase):

    def __init__(self):

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        image = cv2.flip(image, 1)

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                row = []

                for lm in hand_landmarks.landmark:

                    row.extend([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

                X = np.array(row).reshape(1, -1)

                prediction = model.predict(
                    X,
                    verbose=0
                )

                confidence = float(
                    np.max(prediction)
                )

                class_id = int(
                    np.argmax(prediction)
                )

                sign = encoder.inverse_transform(
                    [class_id]
                )[0]

                with shared_state.lock:

                    shared_state.current_sign = sign
                    shared_state.confidence = confidence

                    # AUTO ADD TO SENTENCE
                    current_time = time.time()

                    if (
                        confidence > 0.70 and
                        sign != shared_state.last_sign and
                        current_time -
                        shared_state.last_added_time > 1.5
                    ):

                        shared_state.sentence.append(
                            sign
                        )

                        shared_state.last_sign = sign

                        shared_state.last_added_time = (
                            current_time
                        )

                cv2.putText(
                    image,
                    f"{sign} ({confidence:.2f})",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )

# =====================================
# UI
# =====================================

col1, col2 = st.columns([2,1])

with col1:

    webrtc_streamer(
        key="sign-language",
        video_processor_factory=SignProcessor,
        media_stream_constraints={
            "video": {
                "width": 640,
                "height": 480
            },
            "audio": False
        }
    )

with col2:

    st.subheader("🔍 Current Sign")

    current_sign_placeholder = st.empty()

    st.subheader("📝 Sentence")

    sentence_placeholder = st.empty()

    if st.button("🗑 Clear Sentence"):

        with shared_state.lock:

            shared_state.sentence = []
            shared_state.last_sign = ""

# =====================================
# LIVE UPDATE
# =====================================

while True:

    with shared_state.lock:

        current_sign_placeholder.success(
            f"{shared_state.current_sign} "
            f"({shared_state.confidence:.2f})"
        )

        sentence_placeholder.info(
            " ".join(shared_state.sentence)
        )

    time.sleep(0.5)