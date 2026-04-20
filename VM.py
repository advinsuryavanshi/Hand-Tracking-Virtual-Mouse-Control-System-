# Hand Tracking Virtual Mouse Control System
# This program uses computer vision and hand gesture recognition to control mouse cursor,
# perform clicks, scrolling, and take screenshots using MediaPipe and OpenCV.

# Import required libraries
import cv2  # OpenCV for computer vision and webcam handling
import mediapipe as mp  # MediaPipe for hand tracking
import pyautogui  # PyAutoGUI for mouse control
import numpy as np  # NumPy for numerical operations
import time  # Time module for delays
import math  # Math module for calculations
from util import get_angle, get_distance  # Custom utility functions

# Initialize MediaPipe Hands solution for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

click_start_time = False
click_times = []
click_cooldown = 0.5  # Cooldown time in seconds to prevent multiple clicks
scroll_mode = False
scroll_start_time = False
freeze_cursor = False  # Flag to prevent cursor movement during click gestures
screenshot_cooldown = 2  # Cooldown time in seconds for taking screenshots
last_screenshot_time = 0  # Timestamp of the last screenshot taken



# Get screen dimensions for mouse control mapping
screen_width, screen_height = pyautogui.size()
print("\n hand mouse is running... ")
prev_screen_x, prev_screen_y = 0, 0

# Initialize webcam capture
cap= cv2.VideoCapture(0)
# Check if camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Main processing loop
while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB for MediaPipe processing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame for hand landmarks
    results = hands.process(rgb)
    
    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ===== GESTURE RECOGNITION AND MOUSE CONTROL =====
            # Extract finger tip landmarks for gesture recognition
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]

            # Determine which fingers are extended (1 = extended, 0 = closed)
            # Checks if fingertip is above the middle joint (extended) or below (closed)
            fingers=[
                1 if hand_landmarks.landmark[i].y < hand_landmarks.landmark[i-2].y else 0
                for i in [8, 12, 16, 20]  # Index, middle, ring, pinky fingertips
            ]

            # Calculate distance between thumb and index finger for click detection
            dist=math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)   
            if dist < 0.06:  # Threshold for click gesture (pinch detection)
                if not freeze_cursor:
                    freeze_cursor = True
                    click_times.append(time.time())

                    # Detect double click (two clicks within 0.4 seconds)
                    if len(click_times) >= 2 and (click_times[-1] - click_times[-2]) < 0.4:
                        click_times.pop(0)  # Remove the older click time
                        pyautogui.doubleClick()
                        cv2.putText(frame, 'Double Click', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (34, 255, 255), 2)
                        click_times=[]
                    else:
                        pyautogui.click()
                        cv2.putText(frame, ' single Click', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 34), 2)
                
            else:
                if freeze_cursor:
                    # Short delay to prevent rapid toggling
                    time.sleep(0.1)
                freeze_cursor = False
            
            # Move cursor when not frozen (not clicking)
            if not freeze_cursor:
                screen_x = int(index_tip.x * screen_width)
                screen_y = int(index_tip.y * screen_height)
                pyautogui.moveTo(screen_x, screen_y,duration=0.1)
                prev_screen_x, prev_screen_y = screen_x, screen_y
            
            # Detect scroll mode: activate when all four fingers (index, middle, ring, pinky) are extended
            if sum(fingers) == 4:
                scroll_mode = True
            else:
                 scroll_mode = False   

            # Handle scrolling gestures when all four fingers are extended
            if scroll_mode:
                if index_tip.y < 0.4:  # Scroll up when index finger is high
                    pyautogui.scroll(40)
                    cv2.putText(frame, 'Scroll Up', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (34, 255, 255), 2)
                elif index_tip.y > 0.6:  # Scroll down when index finger is low
                    pyautogui.scroll(-40)
                    cv2.putText(frame, 'Scroll Down', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (34, 255, 255), 2) 

            # Take screenshot when all fingers are closed (fist gesture)
            if sum(fingers) == 0:
                current_time = time.time()
                if current_time - last_screenshot_time > screenshot_cooldown:
                    pyautogui.screenshot(f'screenshot_{int(current_time)}.png')
                    last_screenshot_time = current_time
                    cv2.putText(frame, 'Screenshot Taken', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (34, 255, 255), 2)
              


    # Display the processed frame with hand tracking overlay
    cv2.imshow('Live Mouse', frame)
    
    # Exit on 'q' key press (IMPORTANT: Click on the OpenCV window first to focus it!)
    # Alternative: Press ESC key or close the window
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 27 is ESC key
        print("Exiting... (q or ESC pressed)")
        break

# Release webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
