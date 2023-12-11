import os
import PIL
import json
import torch
import numpy as np

from typing import Optional, Union, List

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from diffusers.models.controlnet import ControlNetConditioningEmbedding
from diffusers.image_processor import VaeImageProcessor
from diffusers.configuration_utils import register_to_config


def load_genhowto_model(weights_path, device="cpu"):
    with open(os.path.join(weights_path, "GenHowTo_controlnet_config.json")) as file:
        gef_controlnet_config = json.load(file)

    controlnet = ControlNetModel.from_config(gef_controlnet_config, torch_dtype=torch.float32)
    # patch forward function of the ControlNet conditioning embedding
    controlnet.controlnet_cond_embedding.forward = GenHowTo_ControlNetConditioningEmbedding_forward.__get__(
        controlnet.controlnet_cond_embedding, ControlNetConditioningEmbedding)
    # load weights for the ControlNet
    controlnet.load_state_dict(torch.load(os.path.join(weights_path, "GenHowTo_controlnet.pth"), map_location="cpu"))

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2", controlnet=controlnet, torch_dtype=torch.float32)
    # load our fine-tuned weights for the UNet
    pipe.unet.load_state_dict(torch.load(os.path.join(weights_path, "GenHowTo_sdunet.pth"), map_location="cpu"))
    # change image preprocessor to our custom one which uses VAE to preprocess input images
    pipe.control_image_processor = GenHowToControlImagePreprocessor(pipe)
    # our model is trained to predict noise directly - we do not use "v_prediction" used by stabilityai/stable-diffusion-2
    pipe.scheduler.config.prediction_type = "epsilon"
    pipe.scheduler.config["prediction_type"] = "epsilon"

    pipe = pipe.to(device)
    if device == "cpu":
        return pipe

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        print("Failed to enable memory efficient attention, continuing without it.")
    return pipe


class GenHowToControlImagePreprocessor:

    def __init__(self, pipe: StableDiffusionControlNetPipeline):
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=pipe.vae_scale_factor, do_convert_rgb=True, do_normalize=True
        )
        self.vae = pipe.vae
        self.scaling_factor = pipe.vae.config.scaling_factor

    @torch.no_grad()
    def preprocess(
            self,
            image: Union[torch.FloatTensor, PIL.Image.Image, np.ndarray],
            height: Optional[int] = None,
            width: Optional[int] = None,
    ) -> torch.Tensor:
        image = self.image_processor.preprocess(image, height=height, width=width)

        w = self.vae.encoder.conv_in.weight
        image = image.to(w.dtype).to(w.device)

        posterior = self.vae.encode(image).latent_dist
        z = posterior.mode()
        # z = z * self.scaling_factor  # scaling is done in the inference script
        return z


def GenHowTo_ControlNetConditioningEmbedding_forward(self, conditioning):
    embedding = self.conv_in(conditioning)
    # # In our case, the input processing copies the main UNet, i.e. no activation function here.
    # embedding = F.silu(embedding)

    assert len(self.blocks) == 0
    # # In our case, we do not use any blocks.
    # for block in self.blocks:
    #     embedding = block(embedding)
    #     embedding = F.silu(embedding)

    # # This is our "zero conv," i.e. it was initialized to zeros at the start of fine-tuning.
    embedding = self.conv_out(embedding)

    return embedding


class DDIMSkipScheduler(DDIMScheduler):

    @register_to_config
    def __init__(self,
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_schedule: str = "linear",
                 trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
                 clip_sample: bool = True,
                 set_alpha_to_one: bool = True,
                 steps_offset: int = 0,
                 prediction_type: str = "epsilon",
                 thresholding: bool = False,
                 dynamic_thresholding_ratio: float = 0.995,
                 clip_sample_range: float = 1.0,
                 sample_max_value: float = 1.0,
                 timestep_spacing: str = "leading",
                 rescale_betas_zero_snr: bool = False):
        super().__init__(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            trained_betas,
            clip_sample,
            set_alpha_to_one,
            steps_offset,
            prediction_type,
            thresholding,
            dynamic_thresholding_ratio,
            clip_sample_range,
            sample_max_value,
            timestep_spacing,
            rescale_betas_zero_snr)
        self.num_steps_to_skip = None

    def set_num_steps_to_skip(self, num_steps_to_skip: int, num_inference_steps: int):
        self.num_steps_to_skip = num_steps_to_skip
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        super().set_timesteps(num_inference_steps, device)
        if self.num_steps_to_skip is None:
            return

        if self.num_steps_to_skip >= num_inference_steps:
            raise ValueError(
                f"`self.num_steps_to_skip`: {self.num_steps_to_skip} cannot be larger or equal to "
                f"`num_inference_steps`: {num_inference_steps}."
            )
        if self.config.timestep_spacing != "leading":
            raise ValueError(
                f"`self.config.timestep_spacing`: {self.config.timestep_spacing} must be `leading` "
                f"if `num_steps_to_skip` is not None."
            )
        self.timesteps = self.timesteps[self.num_steps_to_skip:]
