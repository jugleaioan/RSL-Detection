import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("gesture_model.h5")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

sequence = []
sequence_length = 20  # Window size

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            sequence.append(landmarks)

        if len(sequence) >= sequence_length:
            input_sequence = np.array(sequence[-sequence_length:])
            prediction = model.predict(np.expand_dims(input_sequence, axis=0))
            gesture = np.argmax(prediction)
            cv2.putText(image, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Gesture Recognition", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
