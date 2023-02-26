from keyless_keyboard.config import KeylessConfig
from keyless_keyboard.featurizer import featurize, flatten
import pickle
import os
from cvzone.HandTrackingModule import HandDetector
import cv2

F_MODELS = 'models'
KEYS = [None, 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '.', ',', '?', '!', '\'']

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
        # Setup the capture process.
        print("Starting tool...")
        cap = cv2.VideoCapture(0)
        detector = HandDetector(detectionCon=0.9, maxHands=2)
        frame_time = int(round(1000/self.config.fps, 0))
        num_frames = self.config.fps * (self.config.capture_time / 1000)
        frames = []
        run_cap = True

        # Capture data until told to stop.
        while run_cap:
            success, img = cap.read()
            hands, img = detector.findHands(img)
            frames.append(self.__get_data(hands))

            if frames == num_frames:
                score = self.__score(flatten(frames))
                if score:
                    key = KEYS[score]
                    print(key)
                    frames = []
                else:
                    frames = frames[1:]

            cv2.imshow("Hand Data Cap", img)
            k = cv2.waitKey(frame_time)
            if k == 27: run_cap = False

        # Shutdown.
        cap.release()
        cv2.destroyAllWindows()

    def __model_inference(self, x):
        inference = [self.models[i].predict_proba(x)[:,1] > self.config.thresholds[i] for i in range(len(self.models))]
        return inference
    
    def __score(self, x):
        inference = self.__model_inference(x)
        score = sum(inference * self.config.scores)
        return score
    
    def __get_data(self, hands):
        points = [0 for _ in range(3 * self.config.num_hand_points)]
        if hands:
            r_hand = hands[0] if hands[0]['type'] == 'Right' else hands[1] if len(hands) == 2 else None
            if r_hand:
                points = featurize(r_hand["lmList"])
        return points

if __name__ == '__main__':
    keyless_keyboard = KeylessKeyboard()
    keyless_keyboard.start()