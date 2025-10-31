import cv2
import mediapipe as mp
import pickle
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']  # extract the model from the dict

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
    if not ret or frame is None:
        print("Frame not received, retrying...")
        continue

    H, W, _ =frame.shape
    # to create mirror effect, other wise right hand will appear on left side
    frame = cv2.flip(frame, 1)
    # OpenCV captures img in BGR, mediapipe expect RGB
    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # feed img to mediapipe
    result = hands.process(RGB_frame)

    test_data = []

    # check if there's a hand on screen to place landmarks
    if result.multi_hand_landmarks and result.multi_handedness:
        # map features to left/right hands
        left_hand = np.full(63, -1.0)
        right_hand = np.full(63, -1.0)

        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_label = result.multi_handedness[idx].classification[0].label
            wrist_x, wrist_y, wrist_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
            coords = [(lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z) for lm in hand_landmarks.landmark]
            coords_flat = [c for triplet in coords for c in triplet]

            if hand_label == 'Left':
                left_hand = np.array(coords_flat)
            else:
                right_hand = np.array(coords_flat)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
        # combine both hands for prediction
        test_data = np.concatenate([left_hand, right_hand])
    else:
        # no hands detected
        test_data = np.full(126, -1.0)

    # prediction is a list of only 1 element
    prediction = model.predict([np.asarray(test_data)])
    # Draw bounding box around detected hands
    x_ = [lm.x for hand in result.multi_hand_landmarks for lm in hand.landmark] if result.multi_hand_landmarks else []
    y_ = [lm.y for hand in result.multi_hand_landmarks for lm in hand.landmark] if result.multi_hand_landmarks else []

    if x_ and y_:
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