import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize Mediapipe Hand solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    # Normalize using the wrist as the origin
    wrist = landmarks[0]
    normalized = [(lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]) for lm in landmarks]
    return np.array(normalized)

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
    return None

frame_skip = 4

def collect_data(video_path, output_dir, label):
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_count = 0
    data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count > frame_skip:
            landmarks = extract_landmarks(frame)
            if landmarks is not None:  # Ensure landmarks are valid
                data.append((normalize_landmarks(landmarks), label))
                count += 1

        # Optionally display the frame with landmarks
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(data) > 0:
        # Convert to a structured array before saving
        np_data = np.array(data, dtype=object)  # Use dtype=object for mixed data types
        np.save(os.path.join(output_dir, f"{label}_data.npy"), np_data)
        print(f"Saved {count} samples for label '{label}'")
    else:
        print(f"No valid data collected for label '{label}'")

RSLData = f"RSlData2"
collect_data(f"{RSLData}/RSL_A.mov", f"RSLOutput", "A")
collect_data(f"{RSLData}/RSL_A2.mov", f"RSLOutput", "A2")
collect_data(f"{RSLData}/RSL_A3.mov", f"RSLOutput", "A3")
collect_data(f"{RSLData}/RSL_B.mov", f"RSLOutput", "B")
collect_data(f"{RSLData}/RSL_C.mov", f"RSLOutput", "C")
collect_data(f"{RSLData}/RSL_D.mov", f"RSLOutput", "D")
collect_data(f"{RSLData}/RSL_E.mov", f"RSLOutput", "E")
collect_data(f"{RSLData}/RSL_F.mov", f"RSLOutput", "F")
collect_data(f"{RSLData}/RSL_G.mov", f"RSLOutput", "G")
collect_data(f"{RSLData}/RSL_H.mov", f"RSLOutput", "H")
collect_data(f"{RSLData}/RSL_I.mov", f"RSLOutput", "I")
collect_data(f"{RSLData}/RSL_J.mov", f"RSLOutput", "J")
collect_data(f"{RSLData}/RSL_K.mov", f"RSLOutput", "K")
collect_data(f"{RSLData}/RSL_L.mov", f"RSLOutput", "L")
collect_data(f"{RSLData}/RSL_M.mov", f"RSLOutput", "M")
collect_data(f"{RSLData}/RSL_N.mov", f"RSLOutput", "N")
collect_data(f"{RSLData}/RSL_O.mov", f"RSLOutput", "O")
collect_data(f"{RSLData}/RSL_P.mov", f"RSLOutput", "P")
collect_data(f"{RSLData}/RSL_Q.mov", f"RSLOutput", "Q")
collect_data(f"{RSLData}/RSL_R.mov", f"RSLOutput", "R")
collect_data(f"{RSLData}/RSL_S.mov", f"RSLOutput", "S")
collect_data(f"{RSLData}/RSL_S2.mov", f"RSLOutput", "S2")
collect_data(f"{RSLData}/RSL_T.mov", f"RSLOutput", "T")
collect_data(f"{RSLData}/RSL_T2.mov", f"RSLOutput", "T2")
collect_data(f"{RSLData}/RSL_U.mov", f"RSLOutput", "U")
collect_data(f"{RSLData}/RSL_V.mov", f"RSLOutput", "V")
collect_data(f"{RSLData}/RSL_W.mov", f"RSLOutput", "W")
collect_data(f"{RSLData}/RSL_X.mov", f"RSLOutput", "X")
collect_data(f"{RSLData}/RSL_Y.mov", f"RSLOutput", "Y")
collect_data(f"{RSLData}/RSL_Z.mov", f"RSLOutput", "Z")
