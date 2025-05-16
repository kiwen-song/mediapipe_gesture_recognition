from django.shortcuts import render
from django.http import JsonResponse
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import base64
import os

# 加载模型
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_cnn', 'gesture_model_cnn.h5')
model = tf.keras.models.load_model(MODEL_PATH)

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Label mapping (from 0 to 1-10)
LABELS = [str(i) for i in range(1, 11)]

def index(request):
    return render(request, 'recognizer/index.html')

def recognize(request):
    if request.method == 'POST':
        try:
            image_data = request.POST.get('image_data')
            _, encoded_data = image_data.split(',')
            decoded_data = base64.b64decode(encoded_data)
            nparr = np.frombuffer(decoded_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            gesture_name = "No Hand Detected"
            confidence = 0.0
            landmarks_drawn = False

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks_drawn = True

                    data = []
                    for lm in hand_landmarks.landmark:
                        data.extend([lm.x, lm.y, lm.z])

                    if len(data) == 63:
                        prediction = model.predict(np.array([data]))[0]
                        class_id = np.argmax(prediction)
                        gesture_name = LABELS[class_id]
                        confidence = prediction[class_id]
                        break  # Process only one hand

            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            return JsonResponse({'gesture': gesture_name, 'confidence': float(confidence),
                                 'frame': f"data:image/jpeg;base64,{frame_base64}", 'landmarks_drawn': landmarks_drawn})
        except Exception as e:
            print(f"Error processing frame: {e}")
            return JsonResponse({'error': str(e)})
    return JsonResponse({'error': 'Invalid request'})