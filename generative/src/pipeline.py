import numpy as np
import torch
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer


def load_text_encoder(model_id: str, device: str, dtype):
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    text_encoder.eval()
    return tokenizer, text_encoder


def tokenize(prompts, tokenizer):
    if isinstance(prompts, str):
        prompts = [prompts]
    return tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )


@torch.no_grad()
def encode_prompt(prompts, tokenizer, text_encoder):

    tokens = tokenize(prompts, tokenizer)
    input_ids = tokens.input_ids.to(text_encoder.device)
    return text_encoder(input_ids).last_hidden_state
    
    
@torch.no_grad()
def pooled_embeddings(prompts, tokenizer, text_encoder, batch_size: int = 64):
    out = []
    for i in range(0, len(prompts), batch_size):
        ids = tokenize(prompts[i:i + batch_size], tokenizer).input_ids.to(text_encoder.device)
        out.append(text_encoder(ids).pooler_output.float().cpu())
    return torch.cat(out)


def load_diffusion_components(model_id: str, device: str, dtype):
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=dtype
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype
    ).to(device)
    unet.eval()
    vae.eval()
    scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
    return unet, vae, scheduler


@torch.no_grad()
def decode_latents(latents, vae):
    images = vae.decode(latents / vae.config.scaling_factor).sample
    images = (images / 2 + 0.5).clamp(0, 1)                # [-1, 1] -> [0, 1]
    images = images.permute(0, 2, 3, 1).float().cpu().numpy()
    return [Image.fromarray((img * 255).round().astype(np.uint8)) for img in images]


@torch.no_grad()
def generate(
    prompt,
    tokenizer,
    text_encoder,
    unet,
    vae,
    scheduler,
    *,
    negative_prompt="",
    guidance_scale=7.5,
    num_steps=30,
    height=512,
    width=512,
    generator=None,
):
    device, dtype = unet.device, unet.dtype

    cond = encode_prompt(prompt, tokenizer, text_encoder)
    uncond = encode_prompt(negative_prompt, tokenizer, text_encoder)
    text_emb = torch.cat([uncond, cond])

    scheduler.set_timesteps(num_steps, device=device)
    shape = (1, unet.config.in_channels, height // 8, width // 8)
    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        latent_in = scheduler.scale_model_input(torch.cat([latents] * 2), t)
        noise_pred = unet(latent_in, t, encoder_hidden_states=text_emb).sample
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return decode_latents(latents, vae)[0]
