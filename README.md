# GenHowTo: Learning to Generate Actions and State Transformations from Instructional Videos

### [[Project Website :dart:]](https://soczech.github.io/genhowto/)&nbsp;&nbsp;&nbsp;[[Paper :page_with_curl:]](https://arxiv.org/abs/2312.07322)&nbsp;&nbsp;&nbsp;[Code :octocat:]

This repository contrains code for the paper [GenHowTo: Learning to Generate Actions and State Transformations from Instructional Videos](https://arxiv.org/abs/2312.07322).

<img src="https://soczech.github.io/assets/img/GenHowTo.svg" style="width:100%">


## Run the model on your images and prompts
1. **Environment setup**
   - Use provided `Dockerfile` to build the environment (`docker build -t genhowto .`) or install the packages manually (`pip install diffusers==0.18.2 transformers xformers accelerate`).
   - The code was tested with PyTorch 2.0.

2. **Download GenHowTo model weights**
   - Use `download_weights.sh` script or download the [GenHowTo weights](https://data.ciirc.cvut.cz/public/projects/2023GenHowTo/weights/GenHowTo-STATES-96h-v1) manually.

3. **Get predictions**
   - Run the following command to get predictions for your image and prompt.
     ```
     python genhowto.py --weights_path weights/GenHowTo-STATES-96h-v1
                        --input_image path/to/image.jpg
                        --prompt "your prompt"
                        --output_path path/to/output.jpg
                        --num_images 1
                        [--num_steps_to_skip 2]
     ```
   - `--num_steps_to_skip` is the number of steps to skip in the diffusion process.
     The higher the number, the more similar the generated image will be to the input image.


## Citation
```bibtex
@article{soucek2023genhowto,
    title={GenHowTo: Learning to Generate Actions and State Transformations from Instructional Videos},
    author={Sou\v{c}ek, Tom\'{a}\v{s} and Damen, Dima and Wray, Michael and Laptev, Ivan and Sivic, Josef},
    month = {December},
    year = {2023}
}
```


## Acknowledgements
This work was partly supported by the EU Horizon Europe Programme under the project EXA4MIND (No. 101092944) and the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (ID:90140). Part of this work was done within the University of Bristol’s Machine Learning and Computer Vision (MaVi) Summer Research Program 2023. Research at the University of Bristol is supported by EPSRC UMPIRE (EP/T004991/1) and EPSRC PG Visual AI (EP/T028572/1).

