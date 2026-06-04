# model_files/preprocessing.py

import cv2
import numpy as np
import torch
import os
import mediapipe as mp

# --- Configuration ---
NUM_FRAMES = 32
FRAME_SIZE = 112   # Must match training frame_size

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi'}


def preprocess_video(video_path):
    """
    Returns tensor shape:
    (1, 4, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE)

    4 channels = R, G, B, Mask
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Evenly sample frames
    frame_indices = np.linspace(
        0,
        frame_count - 1,
        NUM_FRAMES,
        dtype=int
    )

    frames = []

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            continue

        # Resize full frame (same as training)
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)

        # Face detection
        results = face_detector.process(frame_rgb)

        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * FRAME_SIZE)
            y = int(bbox.ymin * FRAME_SIZE)
            w = int(bbox.width * FRAME_SIZE)
            h = int(bbox.height * FRAME_SIZE)

            # Expand bounding box (30% padding)
            pad_w = int(0.3 * w)
            pad_h = int(0.3 * h)

            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(FRAME_SIZE, x + w + pad_w)
            y2 = min(FRAME_SIZE, y + h + pad_h)

            mask[y1:y2, x1:x2] = 1.0

        # Normalize RGB
        rgb_norm = frame_rgb / 255.0

        # Create 4-channel frame (RGB + mask)
        frame_4ch = np.concatenate(
            [rgb_norm, mask[..., None]],
            axis=2
        )

        frames.append(frame_4ch)

    cap.release()

    if len(frames) < NUM_FRAMES:
        frames += [frames[-1]] * (NUM_FRAMES - len(frames))

    clip = np.stack(frames, axis=0)          # (T, H, W, C)
    clip = clip.transpose(3, 0, 1, 2)       # (C, T, H, W)

    tensor = torch.tensor(clip, dtype=torch.float32)

    return tensor.unsqueeze(0)  # (1, 4, T, H, W)