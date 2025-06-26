import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import numpy as np
import cv2
import base64

st.title("🎵 Detección de dedos con sonidos - PALOLOCO")

# Sonidos
sound_files = [
    "sounds/#fa.wav", "sounds/la.wav", "sounds/re.wav",
    "sounds/#do.wav", "sounds/#sol.wav", "sounds/si.wav",
    "sounds/la.wav", "sounds/re.wav", "sounds/#do.wav", "sounds/#sol.wav",
]

def play_audio(path):
    with open(path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

# Detección de dedos con MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandGestureDetector(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        self.finger_state = [False] * 10

    def is_finger_down(self, landmarks, tip, mcp):
        return landmarks[tip].y > landmarks[mcp].y

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for h, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                dedos_tip = [4, 8, 12, 16, 20]
                dedos_mcp = [2, 5, 9, 13, 17]

                for i in range(5):
                    index = i + h * 5
                    if self.is_finger_down(hand_landmarks.landmark, dedos_tip[i], dedos_mcp[i]):
                        if not self.finger_state[index]:
                            play_audio(sound_files[index])
                            self.finger_state[index] = True
                    else:
                        self.finger_state[index] = False

        return image

st.subheader("📷 Cámara del navegador")
webrtc_streamer(
    key="paloloco",
    video_transformer_factory=HandGestureDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)
