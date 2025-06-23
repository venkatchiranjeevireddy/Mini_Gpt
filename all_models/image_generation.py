from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

prompt = "a futuristic cityscape at sunset"
print(f"Generating image for prompt: '{prompt}'")
image = pipe(prompt).images[0]
image.save("generated_image.png")
print("Image saved as 'generated_image.png'")
