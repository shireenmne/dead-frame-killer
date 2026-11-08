import argparse
import shutil
import subprocess
import sys
import cv2
import numpy as np

parser = argparse.ArgumentParser(description="Remove duplicate/static frames from a video.")
parser.add_argument("video", help="input video path")
parser.add_argument("-o", "--output", default=None, help="output path (default: <input>_clean.mp4)")
parser.add_argument("--mode", choices=["held", "static", "both"], default="both",
                    help="held = short repeats only, static = long static stretches only, both = everything")
parser.add_argument("--threshold", type=float, default=0.05, help="mean pixel change below this = duplicate")
parser.add_argument("--long-run", type=int, default=10, help="runs this many frames or longer count as static stretches")
args = parser.parse_args()

if shutil.which("ffmpeg") is None:
    sys.exit("ffmpeg not found. Install it with: brew install ffmpeg")

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    sys.exit(f"Could not open '{args.video}'. Check the path (quote names with spaces).")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output = args.output or args.video.rsplit(".", 1)[0] + "_clean.mp4"

# --- Pass 1: find duplicate frames (same detector as duplicate_frames.py) ---
is_dup, prev = [], None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    is_dup.append(prev is not None and float(np.mean(cv2.absdiff(gray, prev))) < args.threshold)
    prev = gray

# Group into runs, then decide which runs to remove based on --mode
remove = [False] * len(is_dup)
i = 0
while i < len(is_dup):
    if is_dup[i]:
        j = i
        while j + 1 < len(is_dup) and is_dup[j + 1]:
            j += 1
        length = j - i + 1
        is_long = length >= args.long_run
        if args.mode == "both" or (args.mode == "static" and is_long) or (args.mode == "held" and not is_long):
            for k in range(i, j + 1):
                remove[k] = True
        i = j + 1
    else:
        i += 1

# --- Pass 2: re-read the video and pipe only the kept frames into ffmpeg ---
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ffmpeg = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error",
     "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast", output],
    stdin=subprocess.PIPE)

kept = 0
for n, rm in enumerate(remove):
    ret, frame = cap.read()
    if not ret:
        break
    if not rm:
        ffmpeg.stdin.write(frame.tobytes())
        kept += 1
cap.release()
ffmpeg.stdin.close()
ffmpeg.wait()

total = len(remove)
print(f"Mode: {args.mode} | threshold: {args.threshold}")
print(f"Frames: {total} -> {kept}  (removed {total - kept}, {100 * (total - kept) / total:.0f}%)")
print(f"Duration: {total / fps:.2f}s -> {kept / fps:.2f}s")
print(f"Saved: {output}")