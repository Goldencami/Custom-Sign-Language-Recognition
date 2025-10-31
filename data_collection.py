import os
import cv2
import csv
import mediapipe as mp
import numpy as np

# Camera number im capturing
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)

# folder where we'll save the data
DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

# csv header
# each hand has 21 landmarks, we have two hands
header = []
for hand_id in ['Left', 'Right']:
    for i in range(21):
        header += [f'{hand_id}_x{i}', f'{hand_id}_y{i}', f'{hand_id}_z{i}']

# variables that contain all information
labels = ['thank-you', 'hug', 'sleepy']
dataset_size = 50

if not cap.isOpened():
    print('Error: Could not open camera.')
    exit()

for gesture in labels:
    # welcoming camera frame
    while True:
        ret, frame = cap.read()
        # needs to see frame in order to save data
        if not ret or frame is None:
            print("Frame not received, retrying...")
            continue
        
        # to create mirror effect, other wise right hand will appear on left side
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, 'Press Q to start recording data...', (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('Video', frame)
        # need to put an argument to waitKey() so it waits before continuing the loop
        # if nothing then it will just wait an infinite amount of time for a key and freeze the video
        # & 0xFF allows for masking so we know we're talking about 'q' on all systems
        if cv2.waitKey(1) & 0xFF == ord('q'): # wait 1ms until we press a key to close window
            break

    print('Preparing data collection...')
    data_coor = [] # save landmarks
    counter = 0

    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Frame not received, retrying...")
            continue
        
        # to create mirror effect, other wise right hand will appear on left side
        frame = cv2.flip(frame, 1)
        # OpenCV captures img in BGR, mediapipe expect RGB
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # feed img to mediapipe
        result = hand.process(RGB_frame)

        # initialize empty vectors for both hands
        left_hand = np.zeros(63)
        right_hand = np.zeros(63)

        # check if there's a hand on screen to place landmarks
        # ensures at least one hand was detected and labeled
        if result.multi_hand_landmarks and result.multi_handedness:
            # loops over each detected hand
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):

                hand_label = result.multi_handedness[idx].classification[0].label  # 'Left or 'Right
                # extracts all 63 numbers (x, y, z for each of the 21 landmarks)
                coords = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

                if hand_label == 'Left':
                    left_hand = np.array(coords)
                else:
                    right_hand = np.array(coords)

                # draw/display landmarks on camera
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # combines both vectors
            frame_data = np.concatenate([left_hand, right_hand])
            data_coor.append(frame_data)
            counter += 1

        cv2.putText(frame, gesture, (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('Video', frame)
        # press 'e' to exit early
        if cv2.waitKey(1) & 0xFF == ord('e'): # wait 1ms until we press a key to close window
            break

    print('Saving data...')
    # save collected data
    file_path = os.path.join(DATA_DIR, f'{gesture}.csv')
    # open a csv file
    with open(file_path, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        # Write the header row
        csv_writer.writerows(data_coor)
        

cap.release()  # release the camera
cv2.destroyAllWindows()