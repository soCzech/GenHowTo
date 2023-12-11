import os
import math
import torch
import argparse
import numpy as np
from PIL import Image

from genhowto_utils import load_genhowto_model, DDIMSkipScheduler


def main(args):
    if os.path.exists(args.output_path):
        print(f"{args.output_path} already exists.")
        return

    pipe = load_genhowto_model(args.weights_path, device=args.device)
    pipe.scheduler.set_timesteps(args.num_inference_steps)

    if args.num_steps_to_skip is not None:  # possibly do not start from complete noise
        pipe.scheduler = DDIMSkipScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_num_steps_to_skip(args.num_steps_to_skip, args.num_inference_steps)
        print(f"Skipping first {args.num_steps_to_skip} DDIM steps, i.e., running DDIM from timestep "
              f"{pipe.scheduler.timesteps[0]} to {pipe.scheduler.timesteps[-1]}.")

    image = Image.open(args.input_image).convert("RGB")
    w, h = image.size
    if w > h:
        image = image.crop(((w - h) // 2, 0, (w + h) // 2, h))
    elif h > w:
        image = image.crop((0, (h - w) // 2, w, (h + w) // 2))
    image = image.resize((512, 512))

    # latents must be passed explicitly, otherwise the model generates incorrect shape
    latents = torch.randn((args.num_images, 4, 64, 64))

    if args.num_inference_steps is not None:
        z = pipe.control_image_processor.preprocess(image)
        z = z * pipe.vae.config.scaling_factor
        t = pipe.scheduler.timesteps[0]
        alpha_bar = pipe.scheduler.alphas_cumprod[t].item()
        latents = math.sqrt(alpha_bar) * z + math.sqrt(1. - alpha_bar) * latents.to(z.device)

    output = pipe(
        args.prompt, image,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        latents=latents,
        num_images_per_prompt=args.num_images,
    ).images

    Image.fromarray(
        np.concatenate([np.array(img) for img in [image] + output], axis=1)
    ).save(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output.png")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_steps_to_skip", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=9.0)
    main(parser.parse_args())
