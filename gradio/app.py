import sys
sys.path.append("./")
import gradio as gr
import torch
from ip_adapter.utils import BLOCKS as BLOCKS
import numpy as np
import random
from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
)
from ip_adapter import StyleStudio_Adapter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "h94/IP-Adapter/sdxl_models/image_encoder"
csgo_ckpt ='InstantX/CSGO/csgo_4_32.bin'
pretrained_vae_name_or_path ='madebyollin/sdxl-vae-fp16-fix'
weight_dtype = torch.float16

vae = AutoencoderKL.from_pretrained(pretrained_vae_name_or_path,torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
    vae=vae
)
pipe.enable_vae_tiling()

target_style_blocks = BLOCKS['style']

csgo = StyleStudio_Adapter(
        pipe, image_encoder_path, csgo_ckpt, device, num_style_tokens=32,
        target_style_blocks=target_style_blocks,
        controlnet_adapter=False,
        style_model_resampler=True,

        fuSAttn=True,
        end_fusion=20,
        adainIP=True,
        )

MAX_SEED = np.iinfo(np.int32).max


def get_example():
    case = [
        [
            './assets/style1.jpg',
            "A red apple",
            7.0,
            42,
            10,
         ],
        [
            './assets/style3.jpg',
            "A orange bus",
            7.0,
            42,
            10,
         ],
    ]
    return case

def run_for_examples(style_image_pil, prompt, guidance_scale, seed, end_fusion):
    
    return create_image(
        style_image_pil=style_image_pil,
        prompt=prompt,
        neg_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        guidance_scale=guidance_scale,
        num_inference_steps=50,
        seed=seed,
        end_fusion=end_fusion,
        use_SAttn=True,
        crossModalAdaIN=True,
    )

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def create_image(style_image_pil,
                 prompt,
                 neg_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                 guidance_scale=7,
                 num_inference_steps=50,
                 end_fusion=20,
                 crossModalAdaIN=True,
                 use_SAttn=True,
                 seed=42,
):

    style_image = style_image_pil

    print(seed)
    generator = torch.Generator(device).manual_seed(seed)
    init_latents = torch.randn((1, 4, 128, 128), generator=generator, device="cuda", dtype=torch.float16)
    num_sample=1
    if use_SAttn:
        num_sample=2
        init_latents = init_latents.repeat(num_sample, 1, 1, 1)
    with torch.no_grad():
        images = csgo.generate(pil_style_image=style_image,
                                prompt=prompt,
                                negative_prompt=neg_prompt,
                                height=1024,
                                width=1024,
                                guidance_scale=guidance_scale,
                                num_images_per_prompt=1,
                                num_samples=num_sample,
                                num_inference_steps=num_inference_steps,
                                end_fusion=end_fusion,
                                cross_modal_adain=crossModalAdaIN,
                                use_SAttn=use_SAttn,
                                
                                generator=generator,
                                latents=init_latents,
                                )

    if use_SAttn:
        return [images[1]]
    else:
        return [images[0]]

# Description
title = r"""
<h1 align="center">StyleStudio: Text-Driven Style Transfer with Selective Control of Style Elements</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/Westlake-AGI-Lab/StyleStudio' target='_blank'><b>StyleStudio: Text-Driven Style Transfer with Selective Control of Style Elements</b></a>.<br>
How to use:<br>
1. Upload a style image.
2. <b>Enter your desired prompt</b>.
3. Click the <b>Submit</b> button to begin customization.
4. Share your stylized photo with your friends and enjoy! 😊

Advanced usage:<br>
1. Click advanced options.
2. Choose different guidance and steps.
3. Set the timing for the Teacher Model's participation.
4. Feel free to discontinue using the Cross-Modal AdaIN and the Teacher Model for result comparison.
"""

article = r"""
---
📝 **Tips**
<br>
1. As the value of end_fusion <b>increases</b>, the style gradually diminishes. 
Therefore, it is suggested to set end_fusion to be between <b>1/5 and 1/3</b> of the number of inference steps (num inference steps).
2. If you want to experience style-based CFG, see the details on the <a href="https://github.com/Westlake-AGI-Lab/StyleStudio">GitHub repo</a>.

---
📝 **Citation**
<br>
If our work is helpful for your research or applications, please cite us via:
```bibtex

```
📧 **Contact**
<br>
If you have any questions, please feel free to open an issue or directly reach us out at <b>leimingkun@westlake.edu.cn</b>.
"""

block = gr.Blocks(css="footer {visibility: hidden}").queue(max_size=10, api_open=False)
with block:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Tabs():
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        style_image_pil = gr.Image(label="Style Image", type='pil')

                prompt = gr.Textbox(label="Prompt",
                                    value="A red apple")
                
                neg_prompt = gr.Textbox(label="Negative Prompt",
                                    value="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry")

                with gr.Accordion(open=True, label="Advanced Options"):

                    guidance_scale = gr.Slider(minimum=1, maximum=15.0, step=0.01, value=7.0, label="guidance scale")
                    
                    num_inference_steps = gr.Slider(minimum=5, maximum=200.0, step=1.0, value=50,
                                                    label="num inference steps")
                    
                    end_fusion = gr.Slider(minimum=0, maximum=200, step=1.0, value=20.0, label="end fusion")
                    
                    seed = gr.Slider(minimum=-1000000, maximum=1000000, value=42, step=1, label="Seed Value")
                    
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
                    
                    crossModalAdaIN = gr.Checkbox(label="Cross Modal AdaIN", value=True)
                    use_SAttn = gr.Checkbox(label="Teacher Model", value=True)

                generate_button = gr.Button("Generate Image")

            with gr.Column():
                generated_image = gr.Gallery(label="Generated Image")

        generate_button.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=create_image,
            inputs=[
                    style_image_pil,
                    prompt,
                    neg_prompt,
                    guidance_scale,
                    num_inference_steps,
                    end_fusion,
                    crossModalAdaIN,
                    use_SAttn,
                    seed,],
            outputs=[generated_image])

    gr.Examples(
        examples=get_example(),
        inputs=[style_image_pil, prompt, guidance_scale, seed, end_fusion],
        fn=run_for_examples,
        outputs=[generated_image],
        cache_examples=False,
    )

    gr.Markdown(article)

block.launch(server_name="0.0.0.0", server_port=1234)
