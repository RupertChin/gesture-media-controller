import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, 
                       max_num_hands=2, 
                       min_detection_confidence=0.5, 
                       min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

def recognize_gesture(hand_landmarks, hand_label):
    """Recognize common hand gestures based on landmark positions"""
    
    # Thumb finger
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    # Index finger
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    # Middle finger
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    # Ring finger
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
    # Pinky
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
    # Wrist
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Calculate finger states (extended or not)
    # Make thumb work with both hands
    if hand_label == "Left":
        thumb_extended = thumb_tip.x > thumb_ip.x
    else:
        thumb_extended = thumb_tip.x < thumb_ip.x
    index_extended = index_tip.y < index_dip.y
    middle_extended = middle_tip.y < middle_dip.y
    ring_extended = ring_tip.y < ring_dip.y
    pinky_extended = pinky_tip.y < pinky_dip.y
    
    # Calculate thumb-index distance
    thumb_index_distance = calculate_distance(thumb_tip, index_tip)
    
    # Recognize gestures based on finger states 
    # TODO find a better way to classify the gestures
    if index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "Pointing"
    
    elif index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "Peace sign"
    
    elif index_extended and not middle_extended and not ring_extended and pinky_extended:
        return "Rock on"
    
    elif index_extended and middle_extended and ring_extended and pinky_extended and not thumb_extended:
        return "Four fingers"
    
    elif index_extended and middle_extended and ring_extended and pinky_extended and thumb_extended:
        return "High five"
    
    elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "Fist"
    
    elif thumb_index_distance < 0.07:
        return "OK sign"
    
    else:
        return "Unknown gesture"

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hands
    results = hands.process(image_rgb)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Label hand as left or right
            hand_label = results.multi_handedness[i].classification[0].label
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            
            # Regocnize and display gesture on screen
            gesture = recognize_gesture(hand_landmarks, hand_label)
            cv2.putText(image, f"Gesture: {gesture}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Print gesture to console
            # print(f"Detected gesture: {gesture}")
    
    # Display the resulting frame
    cv2.imshow('Hand Gesture Recognition', image)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Find distance between thumb and index finger tips
    if cv2.waitKey(1) & 0xFF == ord('d'): 
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        print(calculate_distance(thumb_tip, index_tip))

cap.release()
cv2.VideoCapture(0).release()
cv2.destroyAllWindows()