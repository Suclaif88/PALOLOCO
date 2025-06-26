import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import cv2
import base64
import threading
import queue

st.title("🎵 Detección de Dedos con Sonidos - PALOLOCO")

sound_files = [
    "sounds/#fa.wav", "sounds/la.wav", "sounds/re.wav",
    "sounds/#do.wav", "sounds/#sol.wav", "sounds/si.wav",
    "sounds/la.wav", "sounds/re.wav", "sounds/#do.wav", "sounds/#sol.wav"
]

# Cola de sonidos para reproducir en hilo principal
sound_queue = queue.Queue()

def play_audio(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

class HandDetector(VideoTransformerBase):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.finger_state = [False] * 10

    def is_finger_down(self, landmarks, tip, mcp):
        return landmarks[tip].y > landmarks[mcp].y

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for h, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                self.mp_draw.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                tip_ids = [4, 8, 12, 16, 20]
                mcp_ids = [2, 5, 9, 13, 17]

                for i in range(5):
                    idx = i + h * 5
                    if self.is_finger_down(hand_landmarks.landmark, tip_ids[i], mcp_ids[i]):
                        if not self.finger_state[idx]:
                            try:
                                sound_queue.put_nowait(sound_files[idx])
                            except:
                                pass
                            self.finger_state[idx] = True
                    else:
                        self.finger_state[idx] = False

        return img

# Inicia la cámara y procesamiento
webrtc_streamer(
    key="paloloco",
    video_processor_factory=HandDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,  # ✅
)

# Renderizar sonidos desde hilo principal
while not sound_queue.empty():
    play_audio(sound_queue.get_nowait())
