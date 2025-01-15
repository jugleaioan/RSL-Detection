import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

gestures = [
    "A", "A2", "A3", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "S2", "T", "T2", "U", "V", "W", "X", "Y", "Z"
]

# Load model
model = load_model("gesture_recognition_modelv4.h5")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

sequence = []
sequence_length = 15

cap = cv2.VideoCapture(0)


def normalize_landmarks(landmarks):
    # Normalize using the wrist as the origin
    wrist = landmarks[0]
    normalized = [(lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]) for lm in landmarks]
    return np.array(normalized)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract the 21 landmarks (x, y, z) as a 2D array
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            # Normalize the landmarks
            normalized_landmarks = normalize_landmarks(landmarks)

            # Flatten the normalized landmarks into a 1D list
            flattened_landmarks = normalized_landmarks.flatten()

            # Append to the sequence
            sequence.append(flattened_landmarks)

            # Keep only the last `sequence_length` frames
            if len(sequence) > sequence_length:
                sequence = sequence[-sequence_length:]

            # Perform prediction if the sequence length matches the required input length
            if len(sequence) == sequence_length:
                input_sequence = np.array(sequence)  # Shape: (sequence_length, 63)
                input_sequence = input_sequence.reshape(1, sequence_length, 63)  # Add batch dimension
                prediction = model.predict(input_sequence)  # Model expects shape (None, sequence_length, 63)

                if np.max(prediction) > 0.1:  # Confidence threshold
                    gesture = np.argmax(prediction)  # Get the predicted gesture index

                    # Display the predicted gesture on the frame
                    cv2.putText(image, f"Gesture: {gestures[gesture]}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Gesture Recognition", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
