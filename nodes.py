import os
import sys
sys.path.append(os.path.dirname(__file__))
import torch
import folder_paths
from comfy.comfy_types.node_typing import ComfyNodeABC, IO
from datetime import datetime
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np
from transformers import CLIPProcessor

from accelerate import Accelerator
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from omnigen2.utils.img_util import create_collage

NEGATIVE_PROMPT = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"

# Use ComfyUI's models_dir as the base for all model folders
OMNIGEN2_MODEL_DIR = os.path.join(folder_paths.models_dir, "omnigen2")
folder_paths.add_model_folder_path("omnigen2", OMNIGEN2_MODEL_DIR, is_default=True)

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to ComfyUI tensor format [B, H, W, C], float32, [0,1]"""
    img_array = np.array(img).astype(np.float32) / 255.0  # HWC
    if img_array.ndim == 2:  # grayscale
        img_array = np.stack([img_array]*3, axis=-1)
    img_tensor = torch.from_numpy(img_array)  # HWC
    img_tensor = img_tensor.unsqueeze(0)  # [1, H, W, C]
    return img_tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI tensor [B, H, W, C] or [H, W, C] to PIL image"""
    if tensor.ndim == 4:
        tensor = tensor[0]
    img_array = tensor.cpu().numpy()
    img_array = np.clip(img_array, 0, 1)
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

class OmniGen2ModelLoader(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"default": "OmniGen2/OmniGen2", "tooltip": "OmniGen2 model folder name (auto dropdown)"}),
                "dtype": ("STRING", {"default": "bf16", "choices": ["fp32", "fp16", "bf16"], "tooltip": "Precision for inference"}),
                "enable_sequential_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "Sequential CPU offload"}),
                "enable_model_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "Model CPU offload"}),
                "enable_teacache": ("BOOLEAN", {"default": False, "tooltip": "Enable TeaCache"}),
                "teacache_rel_l1_thresh": ("FLOAT", {"default": 0.05, "tooltip": "Relative L1 threshold for teacache."}),
                "enable_taylorseer": ("BOOLEAN", {"default": False, "tooltip": "Enable TaylorSeer"}),
            },
            "optional": {
                "enable_group_offload": ("BOOLEAN", {"default": False, "tooltip": "Group offload (if supported)"}),
            }
        }
    RETURN_TYPES = ("OMNIGEN2MODEL",)
    FUNCTION = "load"
    CATEGORY = "omnigen2"
    DESCRIPTION = "Load OmniGen2 pipeline and weights, support precision and offload options."

    @classmethod
    def INPUT_TYPE_OPTIONS(cls):
        # Auto dropdown for model folders
        model_list = folder_paths.get_filename_list("omnigen2")
        return {"repo_id": {"choices": model_list}}

    def load(self, repo_id, dtype, enable_sequential_cpu_offload, enable_model_cpu_offload, enable_group_offload=False, enable_teacache = False, enable_taylorseer = False, teacache_rel_l1_thresh = 0.05):
        local_name = repo_id.split('/')[-1]
        model_dir = os.path.join(OMNIGEN2_MODEL_DIR, local_name)
        if not os.path.isdir(model_dir):
            # Auto download from HuggingFace if not found locally
            import subprocess
            import sys
            print(f"Model '{repo_id}' not found locally, attempting to download from HuggingFace...")
            dest_dir = model_dir
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir, exist_ok=True)
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=repo_id, local_dir=dest_dir)
            except ImportError:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'huggingface_hub'])
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=repo_id, local_dir=dest_dir)
            print(f"Model '{repo_id}' downloaded to {dest_dir}")
            # After download, check again
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"Model directory '{model_dir}' not found after download.")
        weight_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        accelerator = Accelerator(mixed_precision=dtype)
        pipeline = OmniGen2Pipeline.from_pretrained(
            model_dir,
            processor=CLIPProcessor.from_pretrained(
                model_dir,
                subfolder="processor",
                use_fast=True
            ),
            torch_dtype=weight_dtype,
            trust_remote_code=True,
        )
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            model_dir,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )
        if enable_taylorseer:
            pipeline.enable_taylorseer = True
        elif enable_teacache:
            pipeline.transformer.enable_teacache = True
            pipeline.transformer.teacache_rel_l1_thresh = teacache_rel_l1_thresh

        if enable_sequential_cpu_offload:
            pipeline.enable_sequential_cpu_offload()
        elif enable_model_cpu_offload:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to(accelerator.device)
        return (pipeline,)

class OmniGen2Sampler(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "omnigen2model": ("OMNIGEN2MODEL", {"tooltip": "OmniGen2 pipeline object"}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "Text instruction"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "scheduler": ("STRING", {"default": "euler", "choices": ["euler", "dpmsolver"]}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 150}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0}),
                "image_guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0}),
                "cfg_range_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "cfg_range_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "num_images_per_prompt": ("INT", {"default": 1, "min": 1, "max": 8}),
                "seed": ("INT", {"default": -1, "tooltip": "-1 for random seed"}),
            },
            "optional": {
                "image1": ("IMAGE", {"tooltip": "Input image 1"}),
                "image2": ("IMAGE", {"tooltip": "Input image 2"}),
                "image3": ("IMAGE", {"tooltip": "Input image 3"}),
                "negative_prompt": ("STRING", {"default": NEGATIVE_PROMPT}),
                "max_input_image_side_length": ("INT", {"default": 1024}),
                "max_pixels": ("INT", {"default": 1048576}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "omnigen2"
    DESCRIPTION = "OmniGen2 sampling inference, supports multiple input images, CFG, scheduler, etc."

    def sample(self, omnigen2model, prompt, width, height, scheduler, num_inference_steps, guidance_scale, image_guidance_scale, cfg_range_start, cfg_range_end, num_images_per_prompt, seed, image1=None, image2=None, image3=None, negative_prompt=NEGATIVE_PROMPT, max_input_image_side_length=1024, max_pixels=1048576):
        input_images = [image1, image2, image3]
        input_images = [img for img in input_images if img is not None]
        if input_images:
            processed_images = []
            for img in input_images:
                if isinstance(img, torch.Tensor):
                    pil_img = tensor_to_pil(img)
                    processed_images.append(pil_img)
                else:
                    processed_images.append(img)
            input_images = processed_images
        if len(input_images) == 0:
            input_images = None
        if seed == -1:
            seed = torch.randint(0, 2**16 - 1, (1,)).item()
        accelerator = Accelerator()
        generator = torch.Generator(device=accelerator.device).manual_seed(seed)
        def progress_callback(cur_step, timesteps):
            pass  # Extend for ComfyUI progress callback if needed
        if scheduler == 'euler':
            omnigen2model.scheduler = FlowMatchEulerDiscreteScheduler()
        elif scheduler == 'dpmsolver':
            omnigen2model.scheduler = DPMSolverMultistepScheduler(
                algorithm_type="dpmsolver++",
                solver_type="midpoint",
                solver_order=2,
                prediction_type="flow_prediction",
            )
        results = omnigen2model(
            prompt=prompt,
            input_images=input_images,
            width=width,
            height=height,
            max_input_image_side_length=max_input_image_side_length,
            max_pixels=max_pixels,
            num_inference_steps=num_inference_steps,
            max_sequence_length=1024,
            text_guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            cfg_range=(cfg_range_start, cfg_range_end),
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            output_type="pil",
            step_func=progress_callback,
        )
        output_images = [pil_to_tensor(image) for image in results.images]
        output_image = torch.cat(output_images, dim=0)
        return (output_image,)

NODE_CLASS_MAPPINGS = {
    "OmniGen2ModelLoader": OmniGen2ModelLoader,
    "OmniGen2Sampler": OmniGen2Sampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniGen2ModelLoader": "OmniGen2 Model Loader",
    "OmniGen2Sampler": "OmniGen2 Sampler",
}
