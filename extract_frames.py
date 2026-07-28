import cv2

VIDEO_PATH = "your_video.mp4"
SAMPLE_EVERY_N_SECONDS = 1 

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * SAMPLE_EVERY_N_SECONDS)

print(f"Video FPS: {fps}")
print(f"Sampling every {frame_interval} frames (~{SAMPLE_EVERY_N_SECONDS}s apart)")

frame_count = 0
sampled_count = 0

while cap.isOpened():
    ret, frame = cap.read() 
    if not ret:
        break 

    if frame_count % frame_interval == 0:
        timestamp = frame_count / fps
        filename = f"frame_{sampled_count:04d}_t{timestamp:.1f}s.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        sampled_count += 1

    frame_count += 1

cap.release()
print(f"\nDone. Sampled {sampled_count} frames from {frame_count} total frames.")