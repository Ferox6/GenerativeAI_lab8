from tkinter import Tk, Entry, Button, Label
from diffusers import StableDiffusionPipeline
import torch
from diffusers.utils import load_image, make_image_grid
import tkinter as tk

model_id = "C:\Progs\stable-diffusion-1.5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

root = Tk()
root.title("Image Generation")
root.geometry("600x600")

Label(root, text="Enter your prompt:").pack()
prompt_entry = Entry(root, width=50)
prompt_entry.pack()

Label(root, text="Enter negative prompt:").pack()
negative_prompt_entry = Entry(root, width=50)
negative_prompt_entry.pack()

Label(root, text="Guidance scale (1-15):").pack()
guidance_entry = Entry(root, width=10)
guidance_entry.pack()
guidance_entry.insert(0, "7.5")

def image_generator():
    prompt = prompt_entry.get() or "a photo of a monkey relaxing on a beach"
    negative_prompt = negative_prompt_entry.get() or "low quality, blurry, distorted"
    guidance_scale = float(guidance_entry.get()) if guidance_entry.get().isdigit() else 7.5

    image = pipe(prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale).images[0]
    image.save("generated_image.png")
    print("Image saved as generated_image.png")


def image_based_generator():
    proto_image = "monkey2.jpg"
    init_image = load_image(proto_image)
    prompt = prompt_entry.get() or "cool monkey, wide smile, black glasses, realistic style"
    negative_prompt = negative_prompt_entry.get() or "low quality, blurry, distorted"
    guidance_scale = float(guidance_entry.get()) if guidance_entry.get().isdigit() else 7.5

    grid_image = pipe(prompt, image=init_image, negative_prompt=negative_prompt, guidance_scale=guidance_scale).images[
        0]
    grid_image = make_image_grid([init_image, grid_image], rows=1, cols=2)
    grid_image.save("modified_image.png")
    print("Image saved as modified_image.png")


Button(root, text="Generate Image", command=image_generator).pack(fill=tk.BOTH, expand=True)
Button(root, text="Modify Image", command=image_based_generator).pack(fill=tk.BOTH, expand=True)

root.mainloop()
