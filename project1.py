import cv2
import mediapipe as mp
import pyautogui
import time

def count_fingers(lst):
    cnt = 0
    thresh = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2

    if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh:
        cnt += 1

    if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh:
        cnt += 1

    if (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > thresh:
        cnt += 1

    if (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > thresh:
        cnt += 1

    if (lst.landmark[4].x < lst.landmark[3].x):
        cnt += 1

    return cnt

def detect_thumb_gesture(lst):
    thumb_tip_y = lst.landmark[4].y
    thumb_base_y = lst.landmark[3].y

    # Ensure the gesture detection threshold is appropriate for your use case
    if thumb_tip_y < thumb_base_y - 0.05:  # Thumb is significantly above the base (thumb up)
        return "thumb_up"
    elif thumb_tip_y > thumb_base_y + 0.05:  # Thumb is significantly below the base (thumb down)
        return "thumb_down"
    else:
        return "neutral"

camera_indices = [0, 1, 2, 3, 4, 5]
cap = None

for index in camera_indices:
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Camera index {index} opened successfully.")
        break
    else:
        cap.release()

if not cap or not cap.isOpened():
    print("Error: Could not open webcam with any of the provided indices.")
    exit()

drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands
hand_obj = hands.Hands(max_num_hands=1)

start_init = False
prev = -1
while True:
    end_time = time.time()
    ret, frm = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frm = cv2.flip(frm, 1)
    res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        hand_keyPoints = res.multi_hand_landmarks[0]
        cnt = count_fingers(hand_keyPoints)
        thumb_gesture = detect_thumb_gesture(hand_keyPoints)
        
        print(f"Detected fingers: {cnt}, Thumb gesture: {thumb_gesture}")

        if not(prev == cnt):
            if not(start_init):
                start_time = time.time()
                start_init = True

            elif (end_time - start_time) > 0.2:
                if thumb_gesture == "thumb_up":
                    pyautogui.hotkey('shift', 'n')  # Play next video
                elif thumb_gesture == "thumb_down":
                    pyautogui.hotkey('shift', 'p')  # Play previous video
                elif cnt == 5:
                    pyautogui.hotkey('alt', 'left')  # Play previous video
                elif cnt == 4:
                    pyautogui.press('right')  # Move right
                elif cnt == 3:
                    pyautogui.press("down")  # Move down
                elif cnt == 2: 
                    pyautogui.press("up")  # Move up
                elif cnt == 1:
                    pyautogui.press("space")  # Play/Pause
                elif cnt == 0 and hand_keyPoints.landmark[4].y < hand_keyPoints.landmark[3].y:
                    pyautogui.hotkey('shift', 'n')
                elif cnt == 0 and hand_keyPoints.landmark[4].y > hand_keyPoints.landmark[3].y:
                    pyautogui.hotkey('shift', ' p')  # Previous video shortcut

                prev = cnt
                start_init = False

        drawing.draw_landmarks(frm, hand_keyPoints, hands.HAND_CONNECTIONS)
    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()