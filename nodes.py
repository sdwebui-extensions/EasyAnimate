import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file
import folder_paths
import gc
import torch
from diffusers import (AutoencoderKL, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from .easyanimate.pipeline.pipeline_easyanimate_inpaint import EasyAnimateInpaintPipeline
from .easyanimate.models.autoencoder_magvit import AutoencoderKLMagvit
from .easyanimate.models.transformer3d import Transformer3DModel
from omegaconf import OmegaConf
import os
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from .easyanimate.utils.utils import get_image_to_video_latent
from einops import rearrange

cache_dir = '/stable-diffusion-cache/models'

script_directory = os.path.dirname(os.path.abspath(__file__))

class DownloadAndLoadEasyAnimateModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                    'EasyAnimateV3-XL-2-InP-512x512',
                    'EasyAnimateV3-XL-2-InP-768x768',
                    'EasyAnimateV3-XL-2-InP-960x960'
                    ],
                    ),
            "precision": ([ 'fp16', 'bf16'],
                    {
                    "default": 'fp16'
                    }),
            },
        }

    RETURN_TYPES = ("EASYANIMATESMODEL",)
    RETURN_NAMES = ("easyanimate_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "EasyanimateWrapper"

    def loadmodel(self, model, precision):
        config_path = f"{script_directory}/config/easyanimate_video_slicevae_motion_module_v3.yaml"
        config = OmegaConf.load(config_path)
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        pbar = ProgressBar(3)

        model_path = os.path.join(folder_paths.models_dir, "EasyAnimate", model)
      
        if not os.path.exists(model_path):
            if os.path.exists(cache_dir):
                model_path = os.path.join(cache_dir, 'EasyAnimate', model)
            else:
                print(f"Downloading easyanimate model to: {model_path}")

        if OmegaConf.to_container(config['vae_kwargs'])['enable_magvit']:
            Choosen_AutoencoderKL = AutoencoderKLMagvit
        else:
            Choosen_AutoencoderKL = AutoencoderKL
        vae = Choosen_AutoencoderKL.from_pretrained(
            model_path, 
            subfolder="vae", 
        ).to(dtype)
        pbar.update(1)

        scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder= 'scheduler')
        
        print("Load TRANSFORMER...")
        transformer = Transformer3DModel.from_pretrained(model_path, subfolder= 'transformer', transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs'])).to(dtype).eval()  
        pbar.update(1) 
        if transformer.config.in_channels == 12:
            clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                model_path, subfolder="image_encoder"
            ).to("cuda", dtype)
            clip_image_processor = CLIPImageProcessor.from_pretrained(
                model_path, subfolder="image_encoder"
            )
        else:
            clip_image_encoder = None
            clip_image_processor = None   
        pbar.update(1)

        pipeline = EasyAnimateInpaintPipeline.from_pretrained(
                model_path,
                transformer=transformer,
                scheduler=scheduler,
                vae=vae,
                torch_dtype=dtype,
                clip_image_encoder=clip_image_encoder,
                clip_image_processor=clip_image_processor,
        ).to(device)
    
        easyanimate_model = {
            'pipeline': pipeline, 
            'dtype': dtype
            }

        return (easyanimate_model,)


class EasyAnimateSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "easyanimate_model": ("EASYANIMATESMODEL", ),
                "video_length": ("INT", {"default": 16, "min": 1, "max": 144, "step": 1}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.01}),
                "scheduler": (
                    [ 
                        "Euler",
                        "Euler A",
                        "DPM++",
                        "PNDM",
                        "DDIM",
                    ],
                      {
                    "default": 'Euler'
                    }
                    ),
                "prompt": ("STRING", {"multiline": False, "default": "",}),
                "negative_prompt": ("STRING", {"multiline": False, "default": "",}),
            },
            "optional":{
                "start_img": ("IMAGE",),
                "end_img": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "EasyanimateWrapper"

    def process(self, easyanimate_model, video_length, width, height, seed, steps, cfg, start_img, end_img, scheduler, prompt, negative_prompt, latent=None, denoise_strength=1.0):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        vae_scaling_factor = 0.13025 #SDXL scaling factor

        mm.soft_empty_cache()
        gc.collect()

        pipeline = easyanimate_model['pipeline']

        scheduler_config = {
            "beta_schedule": "linear",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "steps_offset": 1,
        }
        if scheduler == "DPM++":
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == "Euler":
            noise_scheduler = EulerDiscreteScheduler(**scheduler_config)
        elif scheduler == "Euler A":
            noise_scheduler = EulerAncestralDiscreteScheduler(**scheduler_config)
        elif scheduler == "PNDM":
            noise_scheduler = PNDMScheduler(**scheduler_config)
        elif scheduler == "DDIM":
            noise_scheduler = DDIMScheduler(**scheduler_config)

        pipeline.scheduler = noise_scheduler

        generator= torch.Generator(device).manual_seed(seed)

        pipeline.transformer.to(device)

        with torch.no_grad():
            if pipeline.transformer.config.in_channels == 12:
                video_length = int(video_length // pipeline.vae.mini_batch_encoder * pipeline.vae.mini_batch_encoder) if video_length != 1 else 1
                input_video, input_video_mask, clip_image = get_image_to_video_latent(start_img, end_img, video_length=video_length, sample_size=(height, width))
                sample = pipeline(
                    prompt, 
                    video_length = video_length,
                    negative_prompt = negative_prompt,
                    height      = height,
                    width       = width,
                    generator   = generator,
                    guidance_scale = cfg,
                    num_inference_steps = steps,

                    video        = input_video,
                    mask_video   = input_video_mask,
                    clip_image   = clip_image, 
                ).videos
            else:
                sample = pipeline(
                    prompt, 
                    video_length = video_length,
                    negative_prompt = negative_prompt,
                    height      = height,
                    width       = width,
                    generator   = generator,
                    guidance_scale = cfg,
                    num_inference_steps = steps,
                ).videos
            videos = rearrange(sample, "b c t h w -> (b t) h w c")
        return (videos,)   
    

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadEasyAnimateModel": DownloadAndLoadEasyAnimateModel,
    "EasyAnimateSampler": EasyAnimateSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadEasyAnimateModel": "download and load animate model",
    "EasyAnimateSampler": "sample video"
}