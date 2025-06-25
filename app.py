import cv2
import mediapipe as mp
import pygame

mp_manos = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils

pygame.mixer.init()

sounds = [
    pygame.mixer.Sound("sounds/#fa.wav"),   # Pulgar izquierdo
    pygame.mixer.Sound("sounds/la.wav"),    # Índice izquierdo
    pygame.mixer.Sound("sounds/re.wav"),    # Medio izquierdo
    pygame.mixer.Sound("sounds/#do.wav"),   # Anular izquierdo
    pygame.mixer.Sound("sounds/#sol.wav"),  # Meñique izquierdo
    pygame.mixer.Sound("sounds/si.wav"),    # Pulgar derecho
    pygame.mixer.Sound("sounds/la.wav"),    # Índice derecho
    pygame.mixer.Sound("sounds/re.wav"),    # Medio derecho
    pygame.mixer.Sound("sounds/#do.wav"),   # Anular derecho
    pygame.mixer.Sound("sounds/#sol.wav"),  # Meñique derecho
]

def is_finger_down(landmarks, finger_tip, finger_mcp):
    return landmarks[finger_tip].y > landmarks[finger_mcp].y

cap = cv2.VideoCapture(0)

with mp_manos.Hands(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5, 
    max_num_hands=2
) as hands:
    finger_state = [False] * 10 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for h, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                mp_dibujo.draw_landmarks(frame, hand_landmarks, mp_manos.HAND_CONNECTIONS)

                dedos_tip = [4, 8, 12, 16, 20]
                dedos_mcp = [2, 5, 9, 13, 17]

                for i in range(5):
                    finger_index = i + h * 5
                    if is_finger_down(hand_landmarks.landmark, dedos_tip[i], dedos_mcp[i]):
                        if not finger_state[finger_index]:
                            sounds[finger_index].play()
                            finger_state[finger_index] = True
                    else:
                        finger_state[finger_index] = False

        cv2.imshow("Manos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
