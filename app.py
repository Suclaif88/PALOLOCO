import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import cv2
import base64
import numpy as np

st.title("🎵 Detector de Dedos con Sonido - PALOLOCO")

sound_files = [
    "sounds/#fa.wav", "sounds/la.wav", "sounds/re.wav",
    "sounds/#do.wav", "sounds/#sol.wav", "sounds/si.wav",
    "sounds/la.wav", "sounds/re.wav", "sounds/#do.wav", "sounds/#sol.wav"
]

def play_audio_base64(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

class HandTracker(VideoTransformerBase):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.finger_state = [False] * 10

    def is_finger_down(self, landmarks, tip, mcp):
        return landmarks[tip].y > landmarks[mcp].y

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for h, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )
                dedos_tip = [4, 8, 12, 16, 20]
                dedos_mcp = [2, 5, 9, 13, 17]
                for i in range(5):
                    idx = i + h * 5
                    if self.is_finger_down(hand_landmarks.landmark, dedos_tip[i], dedos_mcp[i]):
                        if not self.finger_state[idx]:
                            # Solo reproducir sonido en la app principal
                            st.session_state['play_sound'] = sound_files[idx]
                            self.finger_state[idx] = True
                    else:
                        self.finger_state[idx] = False

        return image

# Activar sonido después del frame
if "play_sound" in st.session_state:
    play_audio_base64(st.session_state['play_sound'])
    del st.session_state['play_sound']

webrtc_streamer(
    key="paloloco",
    video_processor_factory=HandTracker,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
