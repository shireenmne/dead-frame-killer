import cv2
import numpy as np

VIDEO_PATH = "duplicate_test.mp4"
DIFF_THRESHOLD = 0.5  # lower = stricter

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

prev_frame_gray = None
frame_count = 0
duplicate_frames = []

print(f"Scanning {VIDEO_PATH} frame-by-frame for duplicates...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame_gray is not None:
        diff = cv2.absdiff(gray, prev_frame_gray)
        mean_diff = np.mean(diff)

        if mean_diff < DIFF_THRESHOLD:
            timestamp = frame_count / fps
            duplicate_frames.append((frame_count, timestamp, mean_diff))

    prev_frame_gray = gray
    frame_count += 1

cap.release()

print(f"Total frames: {frame_count}")
print(f"Duplicate/near-duplicate frames found: {len(duplicate_frames)}\n")
for fnum, t, diff in duplicate_frames[:20]:
    print(f"Frame {fnum} (t={t:.2f}s) — diff score: {diff:.2f}")