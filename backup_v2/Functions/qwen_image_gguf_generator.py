import os
from pathlib import Path
import torch
from PIL import Image
import time
from typing import Optional

# Import diffusers components with error handling
try:
    from diffusers import DiffusionPipeline, GGUFQuantizationConfig, QwenImageTransformer2DModel
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError as e:
    print(f"Error importing diffusers or transformers: {e}")
    print("Please install the required packages:")
    print("pip install git+https://github.com/huggingface/diffusers")
    print("pip install transformers")
    raise

# Determine the main directory where the script is located
MAIN_DIR = Path(__file__).resolve().parent.parent  # Go up one level from Functions/ to the main directory

# Define the model name and file path
QWEN_IMAGE_MODEL_NAME = "Qwen/Qwen-Image"
QWEN_IMAGE_GGUF_MODEL_PATH = str(MAIN_DIR / "Models" / "Qwen_Image-Q6_K.gguf")  # Absolute path to the model file

# Define the output directory for generated images (relative to the main script directory)
OUTPUT_DIR = str(MAIN_DIR / "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class QwenImageGGUFGenerator:
    def __init__(self, model_path: str = QWEN_IMAGE_GGUF_MODEL_PATH, gpu_memory: str = "24GB"):
        """
        Initialize the Qwen Image Generator with GGUF model
        gpu_memory: "16GB" or "24GB" - specifies the GPU memory to optimize for
        """
        self.gpu_memory = gpu_memory
        self.model_path = model_path
        self.torch_dtype = torch.bfloat16
        
        print(f"Initializing Qwen Image Generator with GGUF model: {model_path}")
        
        # Load the transformer from the GGUF file
        try:
            self.transformer = QwenImageTransformer2DModel.from_single_file(
                self.model_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.torch_dtype),
                torch_dtype=self.torch_dtype,
                config=QWEN_IMAGE_MODEL_NAME,
                subfolder="transformer",
            )
            print("Transformer model loaded successfully from GGUF file.")
        except Exception as e:
            print(f"Error loading transformer from GGUF file: {e}")
            raise e
        
        # For 16GB GPUs, we also need to quantize the text encoder
        if self.gpu_memory == "16GB":
            print("Loading text encoder with 4-bit quantization for 16GB GPU...")
            quantization_config = TransformersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
            )

            self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                QWEN_IMAGE_MODEL_NAME,
                subfolder="text_encoder",
                quantization_config=quantization_config,
                torch_dtype=self.torch_dtype,
            )
            self.text_encoder = self.text_encoder.to("cpu")
            print("Text encoder loaded with 4-bit quantization.")
            
            # Create pipeline with both transformer and text encoder
            self.pipe = DiffusionPipeline.from_pretrained(
                QWEN_IMAGE_MODEL_NAME, 
                transformer=self.transformer, 
                text_encoder=self.text_encoder, 
                torch_dtype=self.torch_dtype
            )
        else:
            # For 24GB+ GPUs
            self.pipe = DiffusionPipeline.from_pretrained(
                QWEN_IMAGE_MODEL_NAME, 
                transformer=self.transformer, 
                torch_dtype=self.torch_dtype
            )
        
        # Enable model CPU offloading to save GPU memory
        self.pipe.enable_model_cpu_offload()
        print("Pipeline initialized and loaded with CPU offloading enabled.")

    def generate_image_from_prompt(self, prompt_text: str, width: int = 1024, height: int = 1024, num_inference_steps: int = 8) -> Optional[Image.Image]:
        """
        Generates an image from a text prompt using the Qwen-Image GGUF model.
        Returns a PIL Image object.
        """
        print(f"\nGenerating image for prompt: '{prompt_text}'")
        print(f"Image dimensions: {width}x{height}, Inference steps: {num_inference_steps}")

        try:
            # Set the generator for reproducible results
            generator = torch.Generator(device="cuda").manual_seed(42)
            
            # Generate the image
            result = self.pipe(
                prompt=prompt_text,
                negative_prompt="",
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=1.0,
                generator=generator,
            )
            
            if result and result.images and len(result.images) > 0:
                print("Image generated successfully.")
                return result.images[0]
            else:
                print("Image generation failed or no image found in response.")
                return None
        except Exception as e:
            print(f"Error during image generation: {e}")
            return None

    def save_and_open_image(self, image: Image.Image, filename_prefix: str = "generated_image"):
        """
        Saves a PIL Image object as a PNG, and opens it in a new window.
        """
        try:
            output_filename = os.path.join(OUTPUT_DIR, f"{filename_prefix}_{int(time.time())}.png")

            image.save(output_filename)
            print(f"Image saved to: {output_filename}")

            # Open the image in a new window (platform-dependent)
            # NOTE: Commenting out the subprocess call that might cause issues in some environments
            # if os.name == 'posix':  # Linux
            #     subprocess.run(['xdg-open', output_filename])
            # elif os.name == 'nt':  # Windows
            #     os.startfile(output_filename)
            # elif hasattr(os, 'uname') and os.uname().sysname == 'Darwin':  # macOS
            #     subprocess.run(['open', output_filename])
            # else:
            #     print("Could not automatically open image. Please open it manually.")

            print("Image saved successfully! You can open it manually at:", output_filename)
            return output_filename
        except Exception as e:
            print(f"Error saving or opening image: {e}")
            return None

# Example Usage:
if __name__ == "__main__":
    # You can specify "16GB" or "24GB" depending on your GPU memory
    generator = QwenImageGGUFGenerator(gpu_memory="24GB")  # Adjust based on your GPU
    user_prompt = input("Enter a prompt for image generation: ")
    if user_prompt:
        pil_image = generator.generate_image_from_prompt(user_prompt)
        if pil_image:
            generator.save_and_open_image(pil_image, filename_prefix=user_prompt.replace(" ", "_")[:50])
    else:
        print("No prompt entered. Exiting.")