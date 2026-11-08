import sys
import cv2
import numpy as np

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "anime_test.mp4"
DIFF_THRESHOLD = 0.05      
LONG_RUN_FRAMES = 10  

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

prev_gray = None
frame_count = 0
is_dup = []   

print(f"Scanning {VIDEO_PATH} frame-by-frame...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        is_dup.append(False)
    else:
        mean_diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
        is_dup.append(mean_diff < DIFF_THRESHOLD)
    prev_gray = gray
    frame_count += 1

cap.release()

runs = []
start = None
for i, dup in enumerate(is_dup):
    if dup and start is None:
        start = i
    elif not dup and start is not None:
        runs.append((start, i - 1))
        start = None
if start is not None:
    runs.append((start, len(is_dup) - 1))

dup_total = sum(b - a + 1 for a, b in runs)
held = [r for r in runs if (r[1] - r[0] + 1) < LONG_RUN_FRAMES]
long_ = [r for r in runs if (r[1] - r[0] + 1) >= LONG_RUN_FRAMES]

print(f"Total frames: {frame_count} ({frame_count / fps:.2f}s at {fps:.2f} fps)")
print(f"Duplicate frames: {dup_total} ({100 * dup_total / frame_count:.0f}% of the video, {dup_total / fps:.2f}s)")
print(f"  Short held-frame runs: {len(held)}  ({sum(b - a + 1 for a, b in held)} frames)")
print(f"  Long static stretches: {len(long_)}  ({sum(b - a + 1 for a, b in long_)} frames)\n")

print("Long static stretches:")
for a, b in long_:
    print(f"  frames {a}-{b}  (t={a / fps:.2f}s to {(b + 1) / fps:.2f}s, {b - a + 1} frames)")