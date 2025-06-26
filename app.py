import cv2
import streamlit as st
import mediapipe as mp
import base64
import time

st.title("🎵 Detección de dedos con sonidos - PALOLOCO")

# Botón para iniciar
if st.button("📷 Activar cámara y detectar dedos"):

    # Sonidos
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

    # Inicializar MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    finger_state = [False] * 10

    def is_finger_down(landmarks, tip, mcp):
        return landmarks[tip].y > landmarks[mcp].y

    start_time = time.time()
    max_duration = 30  # segundos

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("No se pudo acceder a la cámara.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for h, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                dedos_tip = [4, 8, 12, 16, 20]
                dedos_mcp = [2, 5, 9, 13, 17]

                for i in range(5):
                    index = i + h * 5
                    if is_finger_down(hand_landmarks.landmark, dedos_tip[i], dedos_mcp[i]):
                        if not finger_state[index]:
                            play_audio(sound_files[index])
                            finger_state[index] = True
                    else:
                        finger_state[index] = False

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if time.time() - start_time > max_duration:
            st.info("⏳ Tiempo máximo alcanzado. Reinicia para continuar.")
            break

    cap.release()
    hands.close()