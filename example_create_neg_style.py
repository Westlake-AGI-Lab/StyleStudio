from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from PIL import Image
import torch
import numpy as np
import cv2

base_model_path =  "path/to/sdxl"
pretrained_vae_name_or_path = 'path/to/madebyollin_sdxl-vae-fp16-fix'
# https://huggingface.co/xinsir/controlnet-canny-sdxl-1.0
controlnet_path = "path/to/controlnet_sdxl_canny_xinsir"

controlnet_conditioning_scale = 0.5  # recommended for good generalization

controlnet = ControlNetModel.from_pretrained(
    controlnet_path,
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained(pretrained_vae_name_or_path, torch_dtype=torch.float16)

# https://huggingface.co/SG161222/RealVisXL_V4.0
civitai_path = "path/to/realvisxl_v4.0"
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    civitai_path,
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

prompt = "write your prompt"
style_image_path = "path/to/your/style_image"

original_image = cv2.resize(cv2.imread(style_image_path), (512, 512))
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
canny_image = cv2.Canny(gray_image, 100, 200)
canny_image_color = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2BGR)

image = canny_image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

images = pipe(
    prompt=prompt, 
    negative_prompt="(octane render, render, drawing, anime, bad photo, bad photography:1.3), (worst quality, low quality, blurry:1.2), (bad teeth, deformed teeth, deformed lips), (bad anatomy, bad proportions:1.1), (deformed iris, deformed pupils), (deformed eyes, bad eyes), (deformed face, ugly face, bad face), (deformed hands, bad hands, fused fingers), morbid, mutilated, mutation, disfigured", 
    image=image,
    generator=torch.Generator("cuda").manual_seed(42),
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    num_inference_steps=50,
    guidance_scale=7.0,
    ).images

images[0].save("neg_style.jpg")

# combined_image = np.concatenate((
#     original_image, 
#     canny_image_color, 
#     cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)),  axis=1)
# cv2.imwrite(f"./test.jpg", combined_image)

print("final")