## Modified from https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py

import torch
from torch import nn
from transformers import CLIPImageProcessor
from PIL import Image
from typing import List
import numpy as np

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.utils import is_torch_xla_available, deprecate
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps, rescale_noise_cfg
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

from utils import instantiate_from_config
from models.attention_processor import AttnProcessor, DualAttnProcessor, DualAttnASProcessor

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class ScalePredictorWithProjection(nn.Module):
    def __init__(self, text_dim=1024, style_dim=1024, scale_predictor_config=None):
        super().__init__()
        out_dim = scale_predictor_config['params']['embedding_dim']
        scale_emb_dim = scale_predictor_config['params']['output_dim']

        self.text_proj = nn.Linear(text_dim, out_dim)
        self.style_proj = nn.Linear(style_dim, out_dim)
        self.scale_predictor = instantiate_from_config(scale_predictor_config)
        self.out_proj = nn.Sequential(
                nn.Linear(scale_emb_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Tanh(),
        )
    
    def forward(self, text_emb, style_emb):
        text_emb = self.text_proj(text_emb)
        style_emb = self.style_proj(style_emb)

        scale_emb = self.scale_predictor(torch.concat([text_emb, style_emb], dim=1))

        return self.out_proj(scale_emb)


class StyleCrafter(nn.Module):
    def __init__(self, image_context_config, scale_predictor_config, text_dim=1024, style_dim=1024, pretrained=None,
                 scale=1.0, use_norm=False):
        super().__init__()
        self.image_context_model = instantiate_from_config(image_context_config)

        self.scale_predictor = ScalePredictorWithProjection(text_dim=text_dim, style_dim=style_dim, 
                                                            scale_predictor_config=scale_predictor_config)
        self.scale = scale
        self.use_norm = use_norm
        # self.pretrained = pretrained
        self.num_tokens = image_context_config['params']['num_queries']
        
        self.unet = None

    def get_trainable_parameters(self):
        self.unet.requires_grad_(False)

        self.image_context_model.requires_grad_(True)
        self.scale_predictor.requires_grad_(True)
        self.kv_attn_layers.requires_grad_(True)

        for p in self.parameters():
            if p.requires_grad:
                # copied from https://github.com/huggingface/diffusers/blob/33c5d125cb1d25fdcfc04ce450f6d51777bcfa3a/src/diffusers/training_utils.py#L146
                p.data = p.to(torch.float32)

        trainable_parameters = [p for p in self.parameters() if p.requires_grad]

        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters())}")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_parameters)}")
        
        return trainable_parameters

    def create_cross_attention_adapter(self, unet, num_inference_steps=50, end_fusion=10):
        dual_attn_processor = {}
        unet_sd = unet.state_dict()

        cross_attn_idx = 0
        for idx, name in enumerate(unet.attn_processors.keys()):
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            if cross_attention_dim is None:
                dual_attn_processor[name] = AttnProcessor(
                    num_inference_steps=num_inference_steps,
                    end_fusion=end_fusion,
                    name=name,
                )
            else:
                layer_name = name.split(".processor")[0]
                kv_state_dicts = {
                    'k': {'weight': unet_sd[layer_name + ".to_k.weight"]},
                    'v': {'weight': unet_sd[layer_name + ".to_k.weight"]},
                }
                dual_attn_processor[name] = DualAttnASProcessor(
                    hidden_size=hidden_size, 
                    cross_attention_dim=cross_attention_dim, 
                    state_dict=kv_state_dicts,
                    scale=self.scale,
                    use_norm=self.use_norm,
                    num_tokens=self.num_tokens,
                    layer_idx=cross_attn_idx,
                    )
                cross_attn_idx += 1

        unet.set_attn_processor(dual_attn_processor)
        self.add_module('kv_attn_layers', nn.ModuleList(unet.attn_processors.values()))
        # raise NotImplementedError("Need to implement the rest of the function")

    def set_scale(self, scale):
        if scale != self.scale:
            self.scale = scale
            for layer in self.kv_attn_layers:
                if isinstance(layer, DualAttnASProcessor):
                    # print(f"Setting scale from {layer.scale} to {scale}")
                    layer.scale = scale

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds):
        style_context = self.image_context_model(image_embeds)
        scale_scalar = self.scale_predictor(encoder_hidden_states, style_context)
        encoder_hidden_states = torch.cat([encoder_hidden_states, style_context], dim=1)
        cross_attention_kwargs = {'scale_scalar': scale_scalar}
        # Predict the noise residual
        import pdb
        pdb.set_trace()
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_cond_kwargs,
                               cross_attention_kwargs=cross_attention_kwargs).sample
        return noise_pred
    

def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        style_image=None,
        style_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
        raise ValueError(
            f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
            f" {type(callback_steps)}."
        )

    if callback_on_step_end_tensor_inputs is not None and not all(
        k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
    ):
        raise ValueError(
            f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
        )

    if prompt is not None and prompt_embeds is not None:
        raise ValueError(
            f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
            " only forward one of the two."
        )
    elif prompt_2 is not None and prompt_embeds is not None:
        raise ValueError(
            f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
            " only forward one of the two."
        )
    elif prompt is None and prompt_embeds is None:
        raise ValueError(
            "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
        )
    elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
        raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
        raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

    if negative_prompt is not None and negative_prompt_embeds is not None:
        raise ValueError(
            f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
            f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
        )
    elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
        raise ValueError(
            f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
            f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
        )

    # if prompt_embeds is not None and negative_prompt_embeds is not None:
    #     if prompt_embeds.shape != negative_prompt_embeds.shape:
    #         raise ValueError(
    #             "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
    #             f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
    #             f" {negative_prompt_embeds.shape}."
    #         )

    if prompt_embeds is not None and pooled_prompt_embeds is None:
        raise ValueError(
            "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
        )

    if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
        raise ValueError(
            "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
        )

    if style_image is not None and style_image_embeds is not None:
        raise ValueError(
            "Provide either `style_image` or `style_image_embeds`. Cannot leave both `style_image` and `style_image_embeds` defined."
        )

    if style_image_embeds is not None:
        if not isinstance(style_image_embeds, list):
            raise ValueError(
                f"`style_image_embeds` has to be of type `list` but is {type(style_image_embeds)}"
            )
        elif style_image_embeds[0].ndim not in [3, 4]:
            raise ValueError(
                f"`style_image_embeds` has to be a list of 3D or 4D tensors but is {style_image_embeds[0].ndim}D"
            )


class StyleCrafterInference():
    def __init__(self, sd_pipe, image_encoder, style_crafter, device='cuda'):
        self.pipe = sd_pipe
        self.image_encoder = image_encoder
        self.style_crafter = style_crafter

        self.device = device
        self.pipe.to(device, torch.float16)
        self.image_encoder.to(device, torch.float16)
        self.style_crafter.to(device, torch.float16)
        self.clip_image_processor = CLIPImageProcessor()

        self.pipe.check_inputs = lambda *args, **kwargs: check_inputs(self.pipe, *args, **kwargs)
        # print(self.pipe.__call__)
        self.pipe.__call__ = lambda *args, **kwargs: pipe_forward(self.pipe, *args, **kwargs)
        # print(self.pipe.__call__)

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.style_crafter.image_context_model(clip_image_embeds)

        return image_prompt_embeds
    
    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        guidance_scale=5.0,
        style_guidance_scale=5.0,
        height=512,
        width=512,
        **kwargs,
    ):
        # self.style_crafter
        if scale != self.style_crafter.scale:
            self.style_crafter.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            # raw_prompt_embeds = prompt_embeds
            scale_scalar = self.style_crafter.scale_predictor(prompt_embeds, image_prompt_embeds)
            # if prompt_embeds.shape[0] == 2:
                # import pdb
                # pdb.set_trace()
                # prompt_embeds[1] = raw_prompt_embeds[1] 
            # print(f"scale_scalar: {torch.mean(scale_scalar)}")
            cross_attention_kwargs = {'scale_scalar': scale_scalar}

            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            # prompt_embeds = prompt_embeds
            negative_prompt_embeds = negative_prompt_embeds
            # print(f"prompt_embeds: {prompt_embeds.shape}")
            # print(f"negative_prompt_embeds: {negative_prompt_embeds.shape}")

            # generator = torch.Generator(self.device).manual_seed(seed)
            
            images = self.pipe.__call__(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=num_inference_steps,
                # generator=generator,
                height=height,
                width=width,
                cross_attention_kwargs=cross_attention_kwargs,
                guidance_scale=guidance_scale,
                style_guidance_scale=style_guidance_scale,
                **kwargs,
            ).images
                # PIL to numpy

                # print(images)
                # print(np.array(images[0]))
                # print(np.array(images[0]).max())

        return images


@torch.no_grad()
def pipe_forward(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: List[int] = None,
    denoising_end: Optional[float] = None,
    guidance_scale: float = 5.0,
    style_guidance_scale: float = 5.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    style_image_embeds: Optional[List[torch.FloatTensor]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    original_size: Optional[Tuple[int, int]] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Optional[Tuple[int, int]] = None,
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_target_size: Optional[Tuple[int, int]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    **kwargs,
):
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            used in both text-encoders
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image. This is set to 1024 by default for the best results.
            Anything below 512 pixels won't work well for
            [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
            and checkpoints that are not specifically fine-tuned on low resolutions.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image. This is set to 1024 by default for the best results.
            Anything below 512 pixels won't work well for
            [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
            and checkpoints that are not specifically fine-tuned on low resolutions.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        timesteps (`List[int]`, *optional*):
            Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
            in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
            passed will be used. Must be in descending order.
        denoising_end (`float`, *optional*):
            When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
            completed before it is intentionally prematurely terminated. As a result, the returned sample will
            still retain a substantial amount of noise as determined by the discrete timesteps selected by the
            scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
            "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
            Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
        guidance_scale (`float`, *optional*, defaults to 5.0):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        negative_prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
            `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
            input argument.
        ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
        style_image_embeds (`List[torch.FloatTensor]`, *optional*):
            Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of IP-adapters.
            Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should contain the negative image embedding
            if `do_classifier_free_guidance` is set to `True`.
            If not provided, embeddings are computed from the `ip_adapter_image` input argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
            of a plain tuple.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
            Guidance rescale factor should fix overexposure when using zero terminal SNR.
        original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
            `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
            explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
        crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
            `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
            `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
            `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
        target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            For most cases, `target_size` should be set to the desired height and width of the generated image. If
            not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
            section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
        negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            To negatively condition the generation process based on a specific image resolution. Part of SDXL's
            micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
            information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
        negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
            To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
            micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
            information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
        negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            To negatively condition the generation process based on a target image resolution. It should be as same
            as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
            information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
        callback_on_step_end (`Callable`, *optional*):
            A function that calls at the end of each denoising steps during the inference. The function is called
            with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
            callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
            `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
        `tuple`. When returning a tuple, the first element is a list with the generated images.
    """

    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    if callback is not None:
        deprecate(
            "callback",
            "1.0.0",
            "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
        )
    if callback_steps is not None:
        deprecate(
            "callback_steps",
            "1.0.0",
            "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
        )

    # 0. Default height and width to unet
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        ip_adapter_image,
        style_image_embeds,
        callback_on_step_end_tensor_inputs,
    )

    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._denoising_end = denoising_end
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # 3. Encode input prompt
    lora_scale = (
        self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
    )

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=self.clip_skip,
    )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    if self.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

    add_time_ids = self._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = self._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    # if self.do_classifier_free_guidance:
    #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    #     add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
    #     add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
    if self.do_classifier_free_guidance:
        negative_prompt_embeds = negative_prompt_embeds.to(device)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device)
        negative_add_time_ids = negative_add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    if ip_adapter_image is not None or style_image_embeds is not None:
        image_embeds = self.prepare_style_image_embeds(
            ip_adapter_image,
            style_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            self.do_classifier_free_guidance,
        )

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    # 8.1 Apply denoising_end
    if (
        self.denoising_end is not None
        and isinstance(self.denoising_end, float)
        and self.denoising_end > 0
        and self.denoising_end < 1
    ):
        discrete_timestep_cutoff = int(
            round(
                self.scheduler.config.num_train_timesteps
                - (self.denoising_end * self.scheduler.config.num_train_timesteps)
            )
        )
        num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
        timesteps = timesteps[:num_inference_steps]

    # 9. Optionally get Guidance Scale Embedding
    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    self._num_timesteps = len(timesteps)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            # latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            if ip_adapter_image is not None or style_image_embeds is not None:
                added_cond_kwargs["image_embeds"] = image_embeds
            noise_pred_text = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                negative_added_cond_kwargs = {"text_embeds": negative_pooled_prompt_embeds, "time_ids": negative_add_time_ids}
                noise_pred_uncond = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=negative_prompt_embeds,
                    timestep_cond=timestep_cond,
                    added_cond_kwargs=negative_added_cond_kwargs,
                    return_dict=False,
                )[0]

                if style_guidance_scale > 0.0 and style_guidance_scale != self.guidance_scale:
                    import pdb
                    pdb.set_trace()
                    noise_pred_uncond_style = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds[:, :77, :],
                        timestep_cond=timestep_cond,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + style_guidance_scale * (noise_pred_text - noise_pred_uncond_style) \
                        + self.guidance_scale * (noise_pred_uncond_style - noise_pred_uncond)
                    # print(f"{style_guidance_scale} || {self.guidance_scale}")
                else:
                    # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = noise_pred_text

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                negative_pooled_prompt_embeds = callback_outputs.pop(
                    "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                )
                add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

            if XLA_AVAILABLE:
                xm.mark_step()

    if not output_type == "latent":
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
        has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
        else:
            latents = latents / self.vae.config.scaling_factor

        image = self.vae.decode(latents, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
    else:
        image = latents

    if not output_type == "latent":
        # apply watermark if available
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return StableDiffusionXLPipelineOutput(images=image)