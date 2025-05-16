import cv2
import mediapipe as mp
import time
import csv
import os

# Set parameters
TOTAL_GESTURES = 10
SAMPLES_PER_GESTURE = 200
WAIT_TIME = 5  # Wait time between gestures (seconds)
CSV_PATH = 'collected_data_cnn/data_cnn.csv'

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Create CSV file and write header
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
with open(CSV_PATH, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    header = [f'{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
    header.append('label')
    csv_writer.writerow(header)

# Open camera
cap = cv2.VideoCapture(0)

for gesture_label in range(1, TOTAL_GESTURES + 1):
    print(f"\nPreparing to collect data for gesture {gesture_label}, please get ready...")
    time.sleep(WAIT_TIME)  # Give time to prepare for the next gesture

    sample_count = 0
    while sample_count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Get 21 landmarks
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])
                data.append(gesture_label)

                # Write to CSV
                with open(CSV_PATH, mode='a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(data)

                sample_count += 1
                cv2.putText(frame, f'Collecting gesture {gesture_label} - {sample_count}/{SAMPLES_PER_GESTURE}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Collect Gesture Data (Press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Collection complete!")
print("All data collection finished, saved to CSV file.")