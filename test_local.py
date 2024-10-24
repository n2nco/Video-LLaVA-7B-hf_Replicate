from PIL import Image
import numpy as np
import av
import time
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
import torch

def print_log(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def predict_video(video_path, prompt="Why is this video funny?", num_frames=8):
    start_time = time.time()
    print_log(f"Starting prediction at {time.strftime('%H:%M:%S')}")

    # Load models
    print_log("Loading models...")
    model_load_start = time.time()
    model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
    processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
    model.eval()
    print_log(f"Models loaded in {time.time() - model_load_start:.2f}s")

    # Process video
    print_log("Processing video...")
    video_start = time.time()
    container = av.open(video_path)
    total_frames = sum(1 for _ in container.decode(video=0))  # Count frames
    container.seek(0)  # Reset video
    
    # Sample frames evenly
    step = max(1, total_frames // num_frames)
    indices = np.array([i for i in range(0, total_frames, step)])[:num_frames]
    print_log(f"Sampling {len(indices)} frames from {total_frames} total frames")
    
    clip = read_video_pyav(container, indices)
    print_log(f"Video processed in {time.time() - video_start:.2f}s")

    # Generate
    print_log("Generating response...")
    gen_start = time.time()
    full_prompt = f"USER: <video>{prompt} ASSISTANT:"
    inputs = processor(text=full_prompt, videos=clip, return_tensors="pt")
    
    with torch.inference_mode():
        generate_ids = model.generate(**inputs, max_length=80)
    
    result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print_log(f"Generation completed in {time.time() - gen_start:.2f}s")

    total_time = time.time() - start_time
    print_log(f"\nTotal processing time: {total_time:.2f}s")
    print_log(f"Result: {result}")
    
    container.close()
    return result

if __name__ == "__main__":
    video_path = "/Users/b/Downloads/prn.mp4"
    result = predict_video(
        video_path=video_path,
        prompt="What is happening in this video?",
        num_frames=8
    )