import cv2
import numpy as np
import base64
from openai import OpenAI

VIDEO_PATH = "test_assets/semantic_test.mp4"
PIXEL_THRESHOLD = 25         
MOTION_THRESHOLD = 0.0005   
MIN_STATIC_SECONDS = 2.0    

client = OpenAI(
    base_url="http://localhost:13305/api/v1",
    api_key="lemonade"
)

def encode_frame(frame):
    success, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

def label_dead_region(frame):
    response = client.chat.completions.create(
        model="Gemma-4-E2B-it-GGUF",
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "In 3-5 words, describe what this static frame shows (e.g. 'black screen', 'frozen shot of a person')."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_frame(frame)}"}}
            ]
        }]
    )
    return response.choices[0].message.content.strip()

def motion_score(gray, prev_gray):
    diff = cv2.absdiff(gray, prev_gray)
    return np.count_nonzero(diff > PIXEL_THRESHOLD) / diff.size

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
min_static_frames = int(MIN_STATIC_SECONDS * fps)

prev_gray = None
frame_count = 0
region_start_frame = None   
region_sample = None       
regions = []            

def close_region(end_frame):
    if region_start_frame is not None and (end_frame - region_start_frame) >= min_static_frames:
        regions.append((region_start_frame / fps, end_frame / fps, region_sample))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = motion_score(gray, prev_gray) if prev_gray is not None else 1.0

    if frame_count % 15 == 0: 
        print(f"t={frame_count / fps:5.2f}s  motion={score:.5f}")

    if score < MOTION_THRESHOLD:
        if region_start_frame is None:
            region_start_frame = frame_count
            region_sample = frame.copy()
    else:
        close_region(frame_count)
        region_start_frame, region_sample = None, None

    prev_gray = gray
    frame_count += 1

close_region(frame_count)  
cap.release()


print(f"\n--- Confirmed dead regions: {len(regions)} ---\n")
for start, end, sample in regions:
    print(f"t={start:.1f}s - t={end:.1f}s  ({end - start:.1f}s)  →  {label_dead_region(sample)}")