from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import json
import os
import keyless_keyboard.featurizer as featurizer
from keyless_keyboard.config import KeylessConfig
import pickle

F_CLEAN_DATA = os.path.join('data', 'clean')
F_MODELS = 'models'
F_METRICS = 'metrics'

def train_all_models():
    keyless_config = KeylessConfig()

    # Get the train and test data.
    print("Loading datasets...")
    x_train, y_train = load_data_file(os.path.join(F_CLEAN_DATA, 'train.json'))
    x_val, y_val = load_data_file(os.path.join(F_CLEAN_DATA, 'validate.json'))
    x_test, y_test = load_data_file(os.path.join(F_CLEAN_DATA, 'test.json'))

    # Train the models.
    if keyless_config.train:
        print("Training models...")
        neutral = train_model(x_train, y_train, x_val, y_val, -1, 'neutral')
        thumb = train_model(x_train, y_train, x_val, y_val, 0, 'thumb')
        index = train_model(x_train, y_train, x_val, y_val, 1, 'index')
        middle = train_model(x_train, y_train, x_val, y_val, 2, 'middle')
        ring = train_model(x_train, y_train, x_val, y_val, 3, 'ring')
        pinky = train_model(x_train, y_train, x_val, y_val, 4, 'pinky')
    else:
        print("Loading pretraned models...")
        neutral = (pickle.load(open(os.path.join(F_MODELS, 'neutral.pkl'), 'rb')), '', 'neutral')
        thumb = (pickle.load(open(os.path.join(F_MODELS, 'thumb.pkl'), 'rb')), '', 'thumb')
        index = (pickle.load(open(os.path.join(F_MODELS, 'index.pkl'), 'rb')), '', 'index')
        middle = (pickle.load(open(os.path.join(F_MODELS, 'middle.pkl'), 'rb')), '', 'middle')
        ring = (pickle.load(open(os.path.join(F_MODELS, 'ring.pkl'), 'rb')), '', 'ring')
        pinky = (pickle.load(open(os.path.join(F_MODELS, 'pinky.pkl'), 'rb')), '', 'pinky')

    # Print curves and run metrics.
    thresholds = keyless_config.thresholds
    data = ""
    metrics = ""
    for i, model in enumerate([neutral, thumb, index, middle, ring, pinky]):
        data += model[1]
        auc, accuracy, prc, rec, f1, support = evaluate_model(x_test, y_test, i-1, model[0], thresholds)
        metrics += f"{model[2]}\n-----------\nAUC: {auc}\nAcc: {accuracy}\nPrecision: {prc}\nRecall: {rec}\nF1: {f1}\nSupport: {support}\n\n"

    os.makedirs(F_METRICS, exist_ok=True)
    if keyless_config.train:
        with open(os.path.join(F_METRICS, 'curves.txt'), 'w') as fp_metrics:
            fp_metrics.write(data)
    if keyless_config.evaluate:
        with open(os.path.join(F_METRICS, 'metrics.txt'), 'w') as fp_metrics:
            fp_metrics.write(metrics)

def load_data_file(fname):
    with open(fname, 'r') as fp_ds:
        data = json.load(fp_ds)
        y = [s['y'] for s in data]
        x = [featurizer.flatten([featurizer.featurize(t) for t in s['x']]) for s in data]
    return x, y

def train_model(x_train, y_train, x_val, y_val, finger, name):
    print(f"Training {name}")
    y_train = [finger in _y for _y in y_train]
    y_val = [finger in _y for _y in y_val]

    # Fit the model and save it.
    model = LogisticRegression()
    model.fit(x_train, y_train)
    pickle.dump(model, open(os.path.join(F_MODELS, name+'.pkl'), 'wb'))

    # ROC & Precision-Recall
    y_val_pred = model.predict_proba(x_val)[:,1]
    fpr, tpr, roc_thrsh = metrics.roc_curve(y_val, y_val_pred)
    auc = metrics.roc_auc_score(y_val, y_val_pred)
    prc, rec, pr_thrsh = metrics.precision_recall_curve(y_val, y_val_pred)
    data = f"{name} - auc: {auc}\n-----------\nROC (fpr, tpr, threshold)"
    for roc in zip(fpr, tpr, roc_thrsh):
        data += f"\n{roc[0]}, {roc[1]}, {roc[2]}"
    data += "\n-----------\nPrecision-Recall (precision, recall, threshold)"
    for prc in zip(prc, rec, pr_thrsh):
        data += f"\n{prc[0]}, {prc[1]}, {prc[2]}"
    data += "\n\n"

    return model, data, name

def evaluate_model(x, y, finger, model, thresholds):
    thrsh = thresholds[finger+1]
    y = [finger in _y for _y in y]
    y_pred = model.predict_proba(x)[:,1]
    auc = metrics.roc_auc_score(y, y_pred)
    y_pred = [yp > thrsh for yp in y_pred]
    accuracy = metrics.accuracy_score(y, y_pred)
    prc, rec, f1, support = metrics.precision_recall_fscore_support(y, y_pred)
    return auc, accuracy, prc, rec, f1, support

if __name__ == '__main__':
    os.makedirs(F_METRICS, exist_ok=True)
    train_all_models()