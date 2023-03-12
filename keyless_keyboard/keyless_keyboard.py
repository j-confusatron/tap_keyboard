from keyless_keyboard.config import KeylessConfig
from keyless_keyboard.featurizer import featurize, flatten
import pickle
import os
from cvzone.HandTrackingModule import HandDetector
import cv2
import time

F_MODELS = 'models'

class KeylessKeyboard(object):

    def __init__(self):
        self.config = KeylessConfig()
        self.neutral = pickle.load(open(os.path.join(F_MODELS, 'neutral.pkl'), 'rb'))
        self.thumb = pickle.load(open(os.path.join(F_MODELS, 'thumb.pkl'), 'rb'))
        self.index = pickle.load(open(os.path.join(F_MODELS, 'index.pkl'), 'rb'))
        self.middle = pickle.load(open(os.path.join(F_MODELS, 'middle.pkl'), 'rb'))
        self.ring = pickle.load(open(os.path.join(F_MODELS, 'ring.pkl'), 'rb'))
        self.pinky = pickle.load(open(os.path.join(F_MODELS, 'pinky.pkl'), 'rb'))
        self.models = [self.neutral, self.thumb, self.index, self.middle, self.ring, self.pinky]

    def start(self):
        print("Starting Keyless Keyboard...")

        # Setup data trackers.
        frames = [[] for _ in range(self.config.num_windows)]
        frame_counter_mod = [i+1 for i in range(self.config.num_windows)]
        agreement_history = [(None, 0) for _ in range(self.config.num_windows)]
        frame_counter = 0
        keypress = ''

        # Setup the capture process.
        cap = cv2.VideoCapture(0)
        detector = HandDetector(detectionCon=0.9, maxHands=2)
        frame_time = int(round(1000/self.config.fps, 0))
        num_frames = int(round(self.config.fps * (self.config.capture_time / 1000), 0))
        timer = 0
        run_cap = True

        # Capture data until told to stop.
        while run_cap:
            # Cap the next frame and detect the presence of a right hand.
            t_start = time.time()
            success, img = cap.read()
            hands, img = detector.findHands(img)
            data_points = self.__get_data(hands)

            # If no right hand is detected, empty out the frame histories and advance to the next frame cap.
            if not data_points:
                frames = [[] for _ in range(self.config.num_windows)]
                agreement_history = [(None, 0) for _ in range(self.config.num_windows)]
                frame_counter = 0

            # Otherwise, loop over each window and test its frame history for motion.
            else:
                for i, f in enumerate(frames):
                    # Append the frame to the current frame collection. If we have enough frames, go to inference.
                    if i % frame_counter_mod[i] == 0:
                        f.append(data_points)
                        if len(f) > num_frames:
                            f.pop(0)
                    if len(f) == num_frames:

                        # Score the current frame history.
                        score = self.__score(flatten(f))

                        # If the score isn't 0, determine the keypress.
                        if score != 0:
                            # Detect a key press and store it in the agreement history.
                            score = 0 if score < 0 else score
                            key = self.config.keys[score]
                            last_key = agreement_history[i]
                            if last_key[0] == key:
                                last_key[1] += 1
                            else:
                                last_key = (key, 1)
                            
                            # Register the key press if he agreement threshold is met.
                            if last_key[1] >= self.config.agreements:
                                keypress = key
                                timer = num_frames // 2
                                print(key)
                                frames = [[] for _ in range(self.config.num_windows)]
                                agreement_history = [(None, 0) for _ in range(self.config.num_windows)]
                                frame_counter = 0
                                break
            
            # For demo purposes, draw the captured image and keypress information.
            cv2.rectangle(img, (0,0), (60,60), (15,15,15), -1)
            cv2.putText(img, keypress, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Keyless Keyboard Demo", img)

            # Sleep for frame time milliseconds, capture keypresses, and test keypresses for 'esc' to quit.
            frame_counter += 1
            t_delta = int(round((time.time() - t_start) * 1000))
            wait_time = max([1, frame_time-t_delta])
            k = cv2.waitKey(wait_time)
            if k == 27: run_cap = False
            if timer > 0:
                timer -= 1
            else:
                keypress = ''

        # Shutdown.
        cap.release()
        cv2.destroyAllWindows()

    def __model_inference(self, x):
        inference = [self.models[i].predict_proba([x])[0,1] > self.config.thresholds[i] for i in range(len(self.models))]
        return inference
    
    def __score(self, x):
        inference = self.__model_inference(x)
        score = sum([inference[i] * self.config.scores[i] for i in range(1, len(self.models))])
        score = score if score else inference[0] * self.config.scores[0]
        if score:
            print(str(score)+" ", end='')
            print(inference)
        return score
    
    def __get_data(self, hands):
        points = None
        if hands:
            r_hand = hands[0] if hands[0]['type'] == 'Right' else hands[1] if len(hands) == 2 else None
            if r_hand:
                points = featurize(r_hand["lmList"])
        return points

if __name__ == '__main__':
    keyless_keyboard = KeylessKeyboard()
    keyless_keyboard.start()