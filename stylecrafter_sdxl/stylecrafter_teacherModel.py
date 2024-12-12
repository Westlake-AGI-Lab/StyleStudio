import argparse
import glob
from PIL import Image
from omegaconf import OmegaConf
from logging import ERROR

import torch
from transformers import CLIPVisionModelWithProjection
from diffusers import StableDiffusionXLPipeline
from diffusers.utils.logging import set_verbosity
set_verbosity(ERROR)

from utils import instantiate_from_config
from models.stylecrafter import StyleCrafterInference
import cv2

def infer(args):
    ## Step 1: Load models from config
    config = OmegaConf.load(args.config)
    model_config = config.model

    sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
        model_config.pretrained_model_name_or_path, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True,
        add_watermarker=False,
    )
    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_config.image_encoder_path)
    style_crafter = instantiate_from_config(model_config.model)
    style_crafter.create_cross_attention_adapter(
        sdxl_pipe.unet, 
        num_inference_steps=config.steps,
        end_fusion=args.end_fusion)
    style_crafter.load_state_dict(torch.load(config.pretrained, map_location="cpu"))
    print("Successfully loaded StyleCrafter-SDXL from", config.pretrained)

    sc_pipe = StyleCrafterInference(sd_pipe=sdxl_pipe, image_encoder=image_encoder, style_crafter=style_crafter, device='cuda')

    ## Step 2: Prepare generator and Init Latents
    seed = 42
    generator = torch.Generator("cuda").manual_seed(seed)
    init_latents = torch.randn((1, 4, 128, 128), generator=generator, device="cuda", dtype=torch.float16)
    init_latents = init_latents.repeat(args.num_samples, 1, 1, 1)
    
    ## Step 3: Infer
    prompt = args.prompt
    style_image = cv2.resize(
        cv2.imread(args.style_path), 
        (512, 512)
    ) 
    style_image = Image.fromarray(cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB))
    images = sc_pipe.generate(
        pil_image=style_image,
        prompt=prompt,
        negative_prompt="",
        num_samples=args.num_samples,
        num_inference_steps=config.steps,
        seed=args.seed,
        scale=args.scale,
        guidance_scale=5.0,
        style_guidance_scale=5.0,
        width=config.width,
        height=config.height,

        generator=generator,
        latents=init_latents,
    )
    images[0].save("test.jpg")
    
    print("final")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/infer/style_crafter_sdxl.yaml")
    parser.add_argument("--style_path", type=str, default="../assets/style1.jpg")
    parser.add_argument("--prompt", type=str, default="A red apple")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--end_fusion", type=int, default=10)

    args = parser.parse_args()

    infer(args)