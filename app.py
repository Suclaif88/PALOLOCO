import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import cv2

st.title("✋ Detección en Tiempo Real de Dedos")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandDetector(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        total_fingers = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                fingers = [8, 12, 16, 20]
                for i in fingers:
                    if landmarks[i].y < landmarks[i - 2].y:
                        total_fingers += 1

        cv2.putText(img, f"Dedos levantados: {total_fingers}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

webrtc_streamer(key="realtime-dedos", video_transformer_factory=HandDetector)
