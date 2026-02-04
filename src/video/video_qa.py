import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import torch
import ffmpeg
from PIL import Image
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

MODEL = "LanguageBind/Video-LLaVA-7B-hf"
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

def extract_frames(video_path, num_frames=8, size=224):
    try:
        out, err = (
            ffmpeg
            .input(video_path)
            .filter("fps", fps=1)
            .filter("scale", size, size)
            .output(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                vframes=num_frames
            )
            .run(capture_stdout=True, capture_stderr=True)
        )
        if err:
            print(err.decode("utf-8", "ignore"))
    except Exception as e:
        print("FFmpeg error:", e)
        return []

    frame_size = 3 * size * size
    frames = []
    for i in range(0, len(out), frame_size):
        chunk = out[i:i + frame_size]
        if len(chunk) < frame_size:
            break
        frames.append(Image.frombytes("RGB", (size, size), chunk))
        if len(frames) >= num_frames:
            break

    return frames

def main():
    if len(sys.argv) < 2:
        print("Usage: python video_qa.py path/to/video.mp4")
        return

    video_file = sys.argv[1]
    print("Device:", DEVICE)

    processor = VideoLlavaProcessor.from_pretrained(MODEL)
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        MODEL,
        torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
        device_map="auto" if DEVICE != "cpu" else None,
    )

    frames = extract_frames(video_file, num_frames=4)
    if not frames:
        print("No frames extracted.")
        return

    while True:
        question = input("\nQuestion> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        # IMPORTANT: video token is REQUIRED
        prompt = f"<video>\n{question}"

        inputs = processor(
            text=prompt,
            videos=frames,
            return_tensors="pt"
        )

        if DEVICE != "cpu":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=24)

        print(processor.batch_decode(ids, skip_special_tokens=True)[0])

if __name__ == "__main__":
    main()