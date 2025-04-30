import cv2
import numpy as np
import joblib
import requests
from urllib.parse import urlparse
import tempfile
import os
from tensorflow.keras.models import model_from_json

def download_video(url):
    try:
        r = requests.get(url, stream=True)
        parsed = urlparse(url)
        ext = os.path.splitext(parsed.path)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        with open(temp_file.name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return temp_file.name
    except Exception as e:
        print(f"Download error: {e}")
        return None

def extract_frames(video_path, max_frames=5):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame = cv2.resize(frame, (128, 128))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_count += 1
        if len(frames) >= max_frames:
            break

    cap.release()
    return frames

def load_model_from_pickle(pkl_path):
    data = joblib.load(pkl_path)
    model = model_from_json(data['model_json'])
    model.set_weights(data['weights'])
    return model

def predict_frame(model, frame):
    frame = np.expand_dims(frame / 255.0, axis=0)
    pred = model.predict(frame)[0][0]
    return pred
