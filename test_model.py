import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

gestures = [
    "A", "A2", "A3", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "S2", "T", "T2", "U", "V", "W", "X", "Y", "Z"
]

# Load model
model = load_model("gesture_recognition_modelv2.h5")

# Initialize Mediapipe
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define sequence length for gesture recognition
sequence_length = 15
sequence = []

# Load the video clip for testing
video_path = 'RSLAlphabet/RSL_Alphabet.mov'
cap = cv2.VideoCapture(video_path)

# Create empty lists for true labels and predictions
y_true = []
y_pred_classes = []

# Define gesture annotations with precise start and end times (in seconds)
annotations = [
    {"gesture": "A", "start": 0.63, "end": 1.26},
    {"gesture": "A2", "start": 1.26, "end": 1.76},
    {"gesture": "A3", "start": 1.76, "end": 2.53},
    {"gesture": "B", "start": 2.53, "end": 3.16},
    {"gesture": "C", "start": 3.16, "end": 4.16},
    {"gesture": "D", "start": 4.16, "end": 4.8},
    {"gesture": "E", "start": 4.8, "end": 5.76},
    {"gesture": "F", "start": 5.76, "end": 6.96},
    {"gesture": "G", "start": 6.96, "end": 8.2},
    {"gesture": "H", "start": 8.2, "end": 9.2},
    {"gesture": "I", "start": 9.2, "end": 10.33},
    {"gesture": "J", "start": 10.33, "end": 11.5},
    {"gesture": "K", "start": 11.5, "end": 12.53},
    {"gesture": "L", "start": 12.53, "end": 13.26},
    {"gesture": "M", "start": 13.26, "end": 14.1},
    {"gesture": "N", "start": 14.1, "end": 14.93},
    {"gesture": "O", "start": 14.93, "end": 15.6},
    {"gesture": "P", "start": 15.6, "end": 16.43},
    {"gesture": "Q", "start": 16.43, "end": 18.2},
    {"gesture": "R", "start": 18.2, "end": 19.16},
    {"gesture": "S", "start": 19.16, "end": 19.76},
    {"gesture": "S2", "start": 19.76, "end": 20.56},
    {"gesture": "T", "start": 20.56, "end": 21.7},
    {"gesture": "T2", "start": 21.7, "end": 22.23},
    {"gesture": "U", "start": 22.23, "end": 23.0},
    {"gesture": "V", "start": 23.0, "end": 24.03},
    {"gesture": "W", "start": 24.03, "end": 24.8},
    {"gesture": "X", "start": 24.8, "end": 25.56},
    {"gesture": "Y", "start": 25.56, "end": 26.16},
    {"gesture": "Z", "start": 26.16, "end": 26.75}
]

gesture_idx = 0
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second of the video


def normalize_landmarks(landmarks):
    wrist = landmarks[0]  # The wrist landmark
    normalized = [(lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z) for lm in landmarks]
    return np.array(normalized)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate current video time
    current_time = frame_count / fps

    # Convert frame for Mediapipe processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            normalized_landmarks = normalize_landmarks(hand_landmarks.landmark)

            sequence.append(normalized_landmarks)

            if len(sequence) > sequence_length:
                sequence = sequence[-sequence_length:]

            if len(sequence) == sequence_length:
                input_sequence = np.array(sequence).reshape(1, sequence_length, 63)
                prediction = model.predict(input_sequence)

                predicted_class = np.argmax(prediction)

                # Check if current time matches the annotation for the gesture
                if gesture_idx < len(annotations):
                    gesture = annotations[gesture_idx]
                    if gesture["start"] <= current_time <= gesture["end"]:
                        y_true.append(gestures.index(gesture["gesture"]))
                        y_pred_classes.append(predicted_class)

                    # Move to the next gesture if the end time is surpassed
                    if current_time > gesture["end"]:
                        gesture_idx += 1

    # Display the predicted gesture
    if len(y_pred_classes) > 0:
        gesture = gestures[y_pred_classes[-1]]
        cv2.putText(image, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the video frame with predictions
    cv2.imshow("Gesture Recognition", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# Calculate confusion matrix and metrics
cm = confusion_matrix(y_true, y_pred_classes)

# Calculate TP, FP, FN, TN for each class
tp = np.diag(cm)  # True Positives are the diagonal elements
fp = np.sum(cm, axis=0) - tp  # False Positives: Column sum - TP
fn = np.sum(cm, axis=1) - tp  # False Negatives: Row sum - TP
tn = np.sum(cm) - (tp + fp + fn)  # True Negatives: Total sum - (TP + FP + FN)

# Print results
for i, gesture in enumerate(gestures):
    print(f"Class: {gesture}")
    print(f"  TP: {tp[i]}, FP: {fp[i]}, FN: {fn[i]}, TN: {tn[i]}")

precision = tp / (tp + fp + 1e-7)
recall = tp / (tp + fn + 1e-7)
f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
specificity = tn / (tn + fp + 1e-7)

for i, gesture in enumerate(gestures):
    print(f"Class: {gesture}")
    print(
        f"  Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}, F1-Score: {f1_score[i]:.2f}, Specificity: {specificity[i]:.2f}")
