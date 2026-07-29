import cv2
import base64
from openai import OpenAI

VIDEO_PATH = "semantic_test.mp4"
SAMPLE_EVERY_N_SECONDS = 1

client = OpenAI(
    base_url="http://localhost:13305/api/v1",
    api_key="lemonade"
)

def encode_frame(frame):
    success, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

def is_dead_frame(base64_image):
    response = client.chat.completions.create(
        model="Gemma-4-E2B-it-GGUF",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Is this frame a black screen, a frozen/stagnant shot, or otherwise 'dead space' with no meaningful visual action? Answer with exactly one word: YES or NO."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ]
    )
    answer = response.choices[0].message.content.strip().upper()
    return "YES" in answer

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * SAMPLE_EVERY_N_SECONDS)

frame_count = 0
raw_results = [] 

print(f"Analyzing {VIDEO_PATH} (FPS: {fps})...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        timestamp = frame_count / fps
        base64_frame = encode_frame(frame)
        dead = is_dead_frame(base64_frame)
        raw_results.append((timestamp, dead))
        print(f"t={timestamp:5.1f}s  →  {'DEAD' if dead else 'live'}")

    frame_count += 1

cap.release()

confirmed_dead = []

for i, (timestamp, dead) in enumerate(raw_results):
    if not dead:
        continue

    prev_dead = raw_results[i - 1][1] if i > 0 else False
    next_dead = raw_results[i + 1][1] if i < len(raw_results) - 1 else False

    if prev_dead or next_dead:
        confirmed_dead.append(timestamp)
    else:
        print(f"Ignoring likely false positive at t={timestamp:.1f}s (isolated flag)")

print(f"\n--- Summary ---")
print(f"Raw flags: {[t for t, d in raw_results if d]}")
print(f"Confirmed dead moments (after filtering): {confirmed_dead}")