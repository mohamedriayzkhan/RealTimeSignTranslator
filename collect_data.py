import cv2
import csv
import os
import time
import mediapipe as mp

# ==========================
# CONFIG
# ==========================

LABEL = input("Enter sign label (e.g. HELLO): ").upper()

DATASET_FILE = "dataset/dataset.csv"
SAVE_INTERVAL = 0.1  # save every 100 ms

os.makedirs("dataset", exist_ok=True)

# ==========================
# MEDIAPIPE
# ==========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==========================
# CAMERA
# ==========================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

sample_count = 0
last_save_time = time.time()

# ==========================
# COLLECT DATA
# ==========================

with open(DATASET_FILE, "a", newline="") as file:

    writer = csv.writer(file)

    while True:

        success, frame = cap.read()

        if not success:
            print("Failed to access webcam")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                current_time = time.time()

                if current_time - last_save_time >= SAVE_INTERVAL:

                    row = []

                    for lm in hand_landmarks.landmark:
                        row.extend([
                            lm.x,
                            lm.y,
                            lm.z
                        ])

                    row.append(LABEL)

                    writer.writerow(row)

                    sample_count += 1

                    last_save_time = current_time

        # Display info

        cv2.putText(
            frame,
            f"Label: {LABEL}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {sample_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Press Q to Quit",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Real-Time Sign Language Dataset Collection",
            frame
        )

        key = cv2.waitKey(1)

        if key & 0xFF == ord("q"):
            break

# ==========================
# CLEANUP
# ==========================

cap.release()
cv2.destroyAllWindows()

print("\n====================")
print(f"Saved {sample_count} samples")
print(f"Label: {LABEL}")
print(f"Dataset: {DATASET_FILE}")
print("====================")