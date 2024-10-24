import os
import sys
import torch
import numpy as np
import av
import time
from typing import Union
from cog import BasePredictor, Input, Path
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from PIL import Image

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

class Predictor:
    def setup(self) -> None:
      start_time = time.time()
      print_log("Starting model setup...")

      os.environ["TRANSFORMERS_CACHE"] = "/src/model_cache"

      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print_log(f"Using device: {self.device}")

      # Print configuration details
      print_log("\nModel Configuration:")
      print_log(f"• Device: {self.device.type}")
      # print_log(f"• Dtype: {torch.bfloat16 if self.device.type == 'cuda' else torch.float32}")
      # print_log(f"• Device Map: {'auto' if self.device.type == 'cuda' else 'None'}")
      # print_log(f"• Quantization: {'8-bit' if self.device.type == 'cuda' else 'None'}")
      print_log(f"• Cache Dir: {os.environ['TRANSFORMERS_CACHE']}")
      # print_log(f"• Offload Folder: /tmp/offload")
      print_log("--------------------\n")

      model_id = "LanguageBind/Video-LLaVA-7B-hf"
      
      try:
          # Load model with minimal settings and BF16 for faster loading
          self.model = VideoLlavaForConditionalGeneration.from_pretrained(
              model_id,
              torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
              device_map="auto" if self.device.type == "cuda" else None,
              # load_in_8bit=True if self.device.type == "cuda" else False,  # 8-bit loading is faster than 4-bit
              low_cpu_mem_usage=True,
              trust_remote_code=True,
              use_safetensors=True
              # offload_folder="/tmp/offload",  # Enable disk offloading for faster boot?
              # local_files_only=True 
          )
          
          if self.device.type != "cuda":
              self.model = self.model.to(self.device)

          if self.device.type == "cuda":
            print('setting memory fraction to 0.95')
            # torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.95)

          # Load processor
          self.processor = VideoLlavaProcessor.from_pretrained(model_id)
          
          self.model.eval()
          torch.cuda.empty_cache()  # Clear any GPU memory
          print_log(f"Model loaded in {time.time() - start_time:.2f}s")

      except Exception as e:
          print_log(f"Error during setup: {str(e)}")
          raise RuntimeError(f"Model setup failed: {str(e)}") from e
    def predict(
        self,
        video: Path = Input(description="Input video file - upload a file or provide a URL (remote or bytes) via API"),
        prompt: str = "What is happening in this video?",
        num_frames: int = 10,  # This is our passed-in frame count
        max_new_tokens: int = 500,
        temperature: float = 0.1,
        top_p: float = 0.9
    ) -> str:
      try:
        predict_start = time.time()
        print_log(f"Starting prediction at {time.strftime('%H:%M:%S')}")

        print("Prompt:", prompt)

        container = av.open(str(video))  # video is already a string now, but str() ensures safety


        # Get the total number of frames in the video
        # Try to get the total number of frames; if it's zero, count frames manually
        total_frames = container.streams.video[0].frames
        if total_frames == 0:
            total_frames = sum(1 for _ in container.decode(video=0))  # Count manually
            container.seek(0)  # Reset after counting


        # Adjust frames_to_use based on available frames in the video
        frames_to_use = min(total_frames, num_frames) if total_frames > 0 else num_frames
        print_log(f"Using {frames_to_use} frames")

        # Simple index calculation, ensuring we do not exceed the available frames
        indices = np.linspace(0, total_frames - 1, frames_to_use, dtype=int)
        print_log(f"Using indices: {indices}")

        # Read the frames from the video
        clip = read_video_pyav(container, indices)
        print_log(f"Extracted {len(clip)} frames")

        full_prompt = f"USER: <video>{prompt} ASSISTANT:"
        inputs = self.processor(
            text=full_prompt,
            videos=clip,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generate_ids = self.model.generate(
                **inputs,
                max_length=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p
            )

        result = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print_log(f"Total prediction time: {time.time() - predict_start:.2f}s")
        result = result.split("ASSISTANT:")[-1].strip()
        # print(result)
        return result

      except Exception as e:
          print_log(f"Error during prediction: {str(e)}")
          raise RuntimeError(f"Prediction failed: {str(e)}") from e
      finally:
          if 'container' in locals():
              container.close()

if __name__ == "__main__":
    video_path = "https://replicate.delivery/pbxt/LqvC79pNyRFxVRr6Y11QAJqwThSMiHDhu48isy4FgIfkd2d6/20241024062008_83770_no-sub_c9341d18-aa4b-422f-87c7-b322608e90a8.mov"
    predictor = Predictor()
    predictor.setup()
    
    # Warm up run (optional but can help stabilize performance)
    try:
        result = predictor.predict(
            video=video_path,  # Changed from video_path=video_path
            prompt="What is happening in this video?",
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=150
        )
        print("Result:", result)
    except Exception as e:
        print(f"Error: {str(e)}")