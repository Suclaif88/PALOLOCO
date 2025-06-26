import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import numpy as np
import base64

st.title("🎵 Detección de dedos con sonidos - PALOLOCO")

sound_files = [
    "sounds/#fa.wav",
    "sounds/la.wav",
    "sounds/re.wav",
    "sounds/#do.wav",
    "sounds/#sol.wav",
    "sounds/si.wav",
    "sounds/la.wav",
    "sounds/re.wav",
    "sounds/#do.wav",
    "sounds/#sol.wav",
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

class HandTracker(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.finger_state = [False] * 10

    def is_finger_down(self, landmarks, tip, mcp):
        return landmarks[tip].y > landmarks[mcp].y

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = img[:, :, ::-1]
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for h, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
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
        return img

webrtc_streamer(key="paloloco", video_processor_factory=HandTracker)
