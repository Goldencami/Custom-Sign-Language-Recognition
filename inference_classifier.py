import cv2
import mediapipe as mp
import pickle
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Camera number im capturing
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, max_num_hands=2)

if not cap.isOpened():
    print('Error: Could not open camera.')
    exit()

while True:
    ret, frame = cap.read()
    H, W,_=frame.shape
    # to create mirror effect, other wise right hand will appear on left side
    frame = cv2.flip(frame, 1)
    # OpenCV captures img in BGR, mediapipe expect RGB
    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # feed img to mediapipe
    result = hands.process(RGB_frame)

    test_data = []
    x_ = []
    y_ = []
    # check if there's a hand on screen to place landmarks
    if result.multi_hand_landmarks:
        # iterating through different detected hands
        for hand_landmarks in result.multi_hand_landmarks:
            # iterate through the landmarks of one hand (testing)
            for point in hand_landmarks.landmark:
                test_data.extend([point.x, point.y, point.z])

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    # pad zeros for missing hand(s) so always 2 hands (126 features)
    while len(test_data) < 126:
        test_data.extend([0.0, 0.0, 0.0])

    # prediction is a list of only 1 element
    prediction = model.predict([np.asarray(test_data)])

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

        # trying to get the corners of the rectangle containing the hand
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0 ,0), 4) # colour and thickness value
        cv2.putText(frame, prediction[0], (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('Video', frame)
    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'): # wait 1ms until we press a key to close window
        break

cap.release()
cv2.destroyAllWindows()