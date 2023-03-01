from cvzone.HandTrackingModule import HandDetector
import cv2
import json
import os
import time
import itertools
from keyless_keyboard.config import KeylessConfig

# Startup the capture process.
print("Starting data capture tool...")
key_config = KeylessConfig()
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.9, maxHands=2)
frame_time = int(round(1000/key_config.fps, 0))
num_frames = int(round(key_config.fps * (key_config.capture_time / 1000), 0))

# Generate the training scenarios.
def product_dict():
    labels = [[0,1],[0,1],[0,1],[0,1],[0,1]]
    for instance in itertools.product(*labels):
        y = [i for i in range(len(instance)) if instance[i]]
        y = [-1] if not y else y
        yield {'x': instance, 'y': y}
scenarios = list(product_dict())

# Track all recorded data here.
f_data = os.path.join('data', f'data_{time.time()}.json')
history = []
cur_n = 0
cur_rec = {'y': scenarios[cur_n]['y'], 'x': []}

# Capture images until told to quit.
run_cap = True
recording = 0
while run_cap:
    # Get the next screen capture and detect hands.
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        r_hand = hands[0] if hands[0]['type'] == 'Right' else hands[1] if len(hands) == 2 else None
        if r_hand and recording:
            cur_rec['x'].append({'pts': r_hand["lmList"], 'cent': r_hand['center'], 'box': r_hand['bbox']})

    # If we're recording, decrement the recording counter.
    # When the counter reaches 0, save the sample and advance to the next.
    if recording:
        recording -= 1
        if not recording:
            print(cur_rec)
            history.append(cur_rec)
            with open(f_data, 'w') as fp_data:
                json.dump(history, fp_data)
            cur_n = cur_n + 1 if cur_n < (len(scenarios)-1) else 0
            cur_rec = {'y': scenarios[cur_n]['y'], 'x': []}

    # Display the screen.
    cv2.rectangle(img, (0,0), (270,60), (0,0,0), -1)
    color = (0,0,255) if recording else (145,145,145)
    cv2.circle(img,(30,30), 20, color, -1)
    x = 60
    for a in scenarios[cur_n]['x']:
        a_ = ('O', (0,255,0)) if a else ('X', (0,0,255))
        cv2.putText(img, a_[0], (x,50), cv2.FONT_HERSHEY_SIMPLEX, 2, a_[1], 2, cv2.LINE_AA)
        x += 40 
    cv2.imshow("Hand Data Cap", img)
    k = cv2.waitKey(frame_time)

    # Handle key presses.
    # esc: quit capturing data and shutdown.
    if k == 27:
        run_cap = False
        
    # space: start recording, or stop recording and save data.
    elif k == 32:
        if recording:
            recording = False
        else:
            recording = num_frames

    # backspace: delete the last entry and cycle back a letter.
    elif k == 8:
        if recording:
            recording = False
        else:
            if len(history) > 0:
                history = history[:-1]
                with open(f_data, 'w') as fp_data:
                    json.dump(history, fp_data)
                cur_n = cur_n - 1 if cur_n > 0 else len(scenarios) - 1
        cur_rec = {'y': scenarios[cur_n]['y'], 'x': []}

# Shutdown.
cap.release()
cv2.destroyAllWindows()