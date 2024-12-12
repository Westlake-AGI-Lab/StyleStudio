import numpy as np
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import torch
from ip_adapter.utils import BLOCKS as BLOCKS
from ip_adapter.utils import controlnet_BLOCKS as controlnet_BLOCKS
import cv2
from PIL import Image
from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
)

from ip_adapter import StyleStudio_Adapter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16

def main(args):
    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    image_encoder_path = "h94/IP-Adapter/sdxl_models/image_encoder"
    csgo_ckpt ='InstantX/CSGO/csgo_4_32.bin'
    pretrained_vae_name_or_path ='madebyollin/sdxl-vae-fp16-fix'

    vae = AutoencoderKL.from_pretrained(pretrained_vae_name_or_path,torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
        vae=vae,
    )
    pipe.enable_vae_tiling()

    target_style_blocks = BLOCKS['style']

    stylestudio = StyleStudio_Adapter(
        pipe, image_encoder_path, csgo_ckpt, device, num_style_tokens=32,
        target_style_blocks=target_style_blocks,
        controlnet_adapter=False,
        style_model_resampler=True,

        fuAttn=args.fuAttn,
        fuSAttn=args.fuSAttn,
        fuIPAttn=args.fuIPAttn,
        fuScale=args.fuScale,
        end_fusion=args.end_fusion,
        adainIP=args.adainIP,
    )

    seed = 42
    generator = torch.Generator("cuda").manual_seed(seed)
    if args.fuAttn or args.fuSAttn or args.fuIPAttn:
        num_sample=2
    else:
        num_sample=1 
    if num_sample == 2:
        init_latents = torch.randn((1, 4, 128, 128), generator=generator, device="cuda", dtype=torch.float16)
        init_latents = init_latents.repeat(num_sample, 1, 1, 1)
        assert torch.equal(init_latents[0], init_latents[1]) is True
        print("use the cross modal adain and teacher model")
    else:
        init_latents = torch.randn((num_sample, 4, 128, 128), generator=generator, device="cuda", dtype=torch.float16)
        print("only use the cross modal adain")
    
    style_path = args.style_path
    prompt = args.prompt

    if os.path.isdir(style_path):
        style_imgs = [img for img in os.listdir(style_path)]
    else:
        style_imgs = [os.path.basename(style_path)]
    final_images = []

    for style_img in style_imgs:
        style_image = cv2.resize(cv2.imread(os.path.join(style_path, style_img)), (512, 512))
        style_image = Image.fromarray(cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB))

        images = stylestudio.generate(
            pil_style_image=style_image,
            prompt=prompt,
            negative_prompt="",
            height=1024,
            width=1024,
            style_scale=1.0,
            guidance_scale=5,
            num_images_per_prompt=1,
            num_samples=num_sample,
            num_inference_steps=args.num_inference_steps,
            
            generator=generator,
            latents=init_latents,
        )
        if args.fuAttn or args.fuSAttn or args.fuIPAttn:
            assert len(images) == 2
            images[1].save("./test.jpg")
            final_images.append(cv2.cvtColor(np.array(images[1]), cv2.COLOR_RGB2BGR))
        else:
            images[0].save("./test.jpg")
            final_images.append(cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR))
    
    cv2.imwrite("concat_test.jpg", cv2.vconcat(final_images))
    print("final")

    
    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fuAttn", action="store_true")
    parser.add_argument("--fuSAttn", action="store_true")
    parser.add_argument("--fuIPAttn", action="store_true")
    parser.add_argument("--fuScale", type=int, default=0)
    parser.add_argument("--end_fusion", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--adainIP", action="store_true")
    parser.add_argument("--prompt", type=str, default="A red apple")
    parser.add_argument("--style_path", type=str, default="assets")
    parser.add_argument("--neg_style_path", type=str, default=None)

    args = parser.parse_args()

    main(args)