import cv2
import mediapipe as mp

# Camera number im capturing
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

if not cap.isOpened():
    print('Error: Could not open camera.')
    exit()

while True:
    success, frame = cap.read()
    if success:
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                print(hand_landmarks)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Capsture Image', frame)
        # need to put an argument to waitKey() so it waits before continuing the loop
        # if nothing then it will just wait an infinite amount of time for a key and freeze the video
        # & 0xFF allows for masking so we know we're talking about 'q' on all systems
        if cv2.waitKey(1) & 0xFF == ord('q'): # wait 1ms until we press a key to close window
            break

cap.release()  # release the camera
cv2.destroyAllWindows()