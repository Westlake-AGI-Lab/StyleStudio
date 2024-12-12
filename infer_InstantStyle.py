import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import cv2
from ip_adapter import IPAdapterXL_cross_modal

device = "cuda"

def main(args):
    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    image_encoder_path = "path/to/your/ipadapter_sdxl/image_encoder"
    ip_ckpt = "path/to/your/ipadapter_sdxl/ip-adapter_sdxl.bin"
    # load SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe.enable_vae_tiling()

    # load ip-adapter
    # target_blocks=["block"] # for original IP-Adapter
    target_blocks=["up_blocks.0.attentions.1"] # for style blocks only
    
    ip_model = IPAdapterXL_cross_modal(
        pipe, image_encoder_path, ip_ckpt, device, target_blocks=target_blocks,
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
    else:
        init_latents = torch.randn((num_sample, 4, 128, 128), generator=generator, device="cuda", dtype=torch.float16)

    prompt = args.prompt
    style_path = args.style_path
    style_image = cv2.resize(cv2.imread(style_path), (512, 512))
    pil_style_img = Image.fromarray(cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB))
    
    neg_style_img_path = None
    if neg_style_img_path is not None:
        print("using neg style image")
        # try the Neg Style CFG
        neg_style_img = cv2.resize(cv2.imread(neg_style_img_path), (512, 512))
        neg_pil_image = Image.fromarray(cv2.cvtColor(neg_style_img, cv2.COLOR_BGR2RGB))
    else:
        neg_pil_image = None

    images = ip_model.generate(
        pil_image=pil_style_img,
        neg_pil_image=neg_pil_image,
        prompt=prompt,
        negative_prompt= "",
        scale=1.0,
        guidance_scale=5,
        num_samples=num_sample,
        num_inference_steps=args.num_inference_steps,
        
        latents=init_latents,
        generator=generator,
    )
    if args.fuAttn or args.fuSAttn or args.fuIPAttn:
        assert len(images) == 2
        images[1].save("./test.jpg")
    else:
        images[0].save("./test.jpg")
        
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
    parser.add_argument("--style_path", type=str, default="assets/style1.jpg")
    
    
    args = parser.parse_args()
    
    main(args)