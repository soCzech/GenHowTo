FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN pip install diffusers==0.18.2 \
                transformers==4.30.2 \
                xformers==0.0.20 \
                accelerate==0.22.0
