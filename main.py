from tkinter import Tk, Entry, Button, Label
from diffusers import StableDiffusionPipeline
import torch
from diffusers.utils import load_image, make_image_grid
import tkinter as tk


model_id = "C:\Progs\stable-diffusion-1.5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


root = Tk()
root.title("Stable Diffusion App")
root.geometry("600x600")


prompt_label = Label(root, text="Enter your prompt:")
prompt_label.pack()
prompt_entry = Entry(root, width=50)
prompt_entry.pack()

def image_generator():
    prompt = prompt_entry.get()
    if not prompt:
        prompt = "a photo of an astronaut riding a horse on the moon"
    image = pipe(prompt).images[0]
    image.save("generated_image.png")
    print("Image saved as generated_image.png")

def image_based_generator():
    proto_image = "monkey2.jpg"
    init_image = load_image(proto_image)
    prompt = prompt_entry.get()
    if not prompt:
        prompt = "cool monkey, wide smile, black glasses, realistic style"
    grid_image = pipe(prompt, image=init_image).images[0]
    grid_image = make_image_grid([init_image, grid_image],rows = 1, cols = 2)
    grid_image.save("modified_image.png")
    print("Image saved as modified_image.png")


show_btn = Button(root, text="Generate Image", command=image_generator)
show_btn.pack(fill=tk.BOTH, expand=True)

cut_btn = Button(root, text="Modify Image", command=image_based_generator)
cut_btn.pack(fill=tk.BOTH, expand=True)

root.mainloop()