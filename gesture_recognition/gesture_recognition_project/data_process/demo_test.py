import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model_cnn/gesture_model_cnn.h5')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Label mapping (from 0 to 1-10)
LABELS = [str(i) for i in range(1, 11)]

# Open the webcam
cap = cv2.VideoCapture(0)

# Set font styles for better aesthetics
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
color = (0, 255, 0)  # Green for text
bg_color = (0, 0, 0)  # Black for background

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for natural view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Check if hand landmarks are found
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Prepare hand landmarks data for prediction
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            if len(data) == 63:
                prediction = model.predict(np.array([data]))[0]
                class_id = np.argmax(prediction)
                gesture_name = LABELS[class_id]
                confidence = prediction[class_id]

                if confidence > 0.8:
                    # Add a nice overlay for the gesture info
                    text = f"Gesture: {gesture_name} | Confidence: {confidence:.2f}"

                    # Draw a semi-transparent background for the text
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                    cv2.rectangle(frame, (10, 10), (text_width + 20, text_height + 20), bg_color, -1)
                    cv2.putText(frame, text, (10, text_height + 10), font, font_scale, color, font_thickness)

    # Display the frame with the overlayed gesture information
    cv2.imshow("Real-time Gesture Recognition", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
