import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import tensorflow as tf
import pyttsx3

# ======================
# PAGE CONFIG
# ======================

st.set_page_config(
    page_title="Real-Time Sign Language Translator",
    layout="wide"
)

st.title("🤟 Real-Time Sign Language Translator")

# ======================
# LOAD MODEL
# ======================

model = tf.keras.models.load_model("models/sign_model.keras")

with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# ======================
# SESSION STATE
# ======================

if "sentence" not in st.session_state:
    st.session_state.sentence = []

if "last_word" not in st.session_state:
    st.session_state.last_word = ""

if "current_word" not in st.session_state:
    st.session_state.current_word = ""

# ======================
# SIDEBAR
# ======================

col1, col2 = st.columns([3, 1])

with col2:

    st.subheader("🔍 Current Sign")

    sign_box = st.empty()

    st.subheader("📝 Sentence")

    sentence_box = st.empty()

    if st.button("🗑 Clear Sentence"):
        st.session_state.sentence = []
        st.session_state.last_word = ""

    if st.button("🔊 Speak Sentence"):

        text = " ".join(st.session_state.sentence)

        if text:

            engine = pyttsx3.init()

            engine.say(text)

            engine.runAndWait()

# ======================
# MEDIAPIPE
# ======================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ======================
# CAMERA
# ======================

run = st.checkbox("Start Camera")

frame_window = col1.image([])

if run:

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            st.error("Cannot access webcam")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        predicted_word = ""

        confidence = 0

        if result.multi_hand_landmarks:

            for hand_landmarks in result.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                landmarks = []

                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                if len(landmarks) == 63:

                    X = np.array(landmarks).reshape(1, -1)

                    prediction = model.predict(
                        X,
                        verbose=0
                    )

                    class_id = np.argmax(prediction)

                    confidence = np.max(prediction)

                    predicted_word = encoder.inverse_transform(
                        [class_id]
                    )[0]

                    cv2.putText(
                        frame,
                        f"{predicted_word} ({confidence:.2f})",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                    # ===================
                    # ADD WORD TO SENTENCE
                    # ===================

                    if confidence > 0.90:

                        if predicted_word != st.session_state.last_word:

                            st.session_state.sentence.append(
                                predicted_word
                            )

                            st.session_state.last_word = predicted_word

                    st.session_state.current_word = predicted_word

        sign_box.success(
            st.session_state.current_word
        )

        sentence_box.info(
            " ".join(st.session_state.sentence)
        )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame_window.image(
            frame,
            channels="RGB",
            use_container_width=True
        )

    cap.release()