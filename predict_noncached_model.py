import os
import sys
import torch
import numpy as np
import av
import time
from typing import Union
from cog import BasePredictor, Input, Path
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

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

      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      model_id = "LanguageBind/Video-LLaVA-7B-hf"
      
      try:
          # Fastest possible loading configuration
          self.model = VideoLlavaForConditionalGeneration.from_pretrained(
              model_id,
              torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
              device_map="auto" if self.device.type == "cuda" else None,
              use_safetensors=True,
              # Removed low_cpu_mem_usage as it can slow initial loading
              trust_remote_code=True
          )
          
          if self.device.type != "cuda":
              self.model = self.model.to(self.device)

          if self.device.type == "cuda":
              torch.cuda.set_per_process_memory_fraction(0.95)

          # Load processor (this is relatively fast)
          self.processor = VideoLlavaProcessor.from_pretrained(model_id)
          
          self.model.eval()
          torch.cuda.empty_cache()

      except Exception as e:
          print_log(f"Error during setup: {str(e)}")
          raise RuntimeError(f"Model setup failed: {str(e)}") from e

    def predict(
        self,
        video: Path = Input(description="Input video file - upload a file or provide a URL (remote or bytes) via API"),
        prompt: str = "What is happening in this video?",
        num_frames: int = 10,
        max_new_tokens: int = 500,
        temperature: float = 0.1,
        top_p: float = 0.9
    ) -> str:
      try:
        predict_start = time.time()
        container = av.open(str(video))

        total_frames = container.streams.video[0].frames
        if total_frames == 0:
            total_frames = sum(1 for _ in container.decode(video=0))
            container.seek(0)

        frames_to_use = min(total_frames, num_frames) if total_frames > 0 else num_frames
        indices = np.linspace(0, total_frames - 1, frames_to_use, dtype=int)
        clip = read_video_pyav(container, indices)

        full_prompt = f"USER: <video>{prompt} ASSISTANT:"
        inputs = self.processor(
            text=full_prompt,
            videos=clip,
            return_tensors="pt"
        )

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

        return result.split("ASSISTANT:")[-1].strip()

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
    
    try:
        result = predictor.predict(
            video=video_path,
            prompt="What is happening in this video?",
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=150
        )
        print("Result:", result)
    except Exception as e:
        print(f"Error: {str(e)}")