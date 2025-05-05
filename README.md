# Boost-and-Skip: A Simple Guidance-Free Diffusion for Minority Generation (ICML 2025)

[Soobin Um*](https://soobin-um.github.io/), [Beomsu Kim*](https://scholar.google.com/citations?user=TofIFUgAAAAJ&hl=en), and [Jong Chul Ye](https://bispl.weebly.com/professor.html)

This repository contains the implementation of the paper "Boost-and-Skip: A Simple Guidance-Free Diffusion for Minority Generation" (ICML 2025).

## 1. Environment setup
We provide a conda environment configuration file to install all the required dependencies. If you don’t have conda installed, you can get it from [here](https://docs.conda.io/en/latest/miniconda.html).

### 1) Clone the repository
```
git clone https://github.com/anonymous7172/BnS
cd BnS
```

### 2) Install dependencies
The code has been tested with the following setup:
- Python 3.11
- PyTorch 2.0.1
- CUDA 11.7

We recommend creating a new conda environment to avoid conflicts with existing installations. You can create a new environment and install the dependencies with the following commands:
```
conda create -n BnS python=3.11.4
conda activate BnS
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
conda install -c conda-forge mpi4py mpich
pip install blobfile
pip install scikit-learn
```


## 2. Download pre-trained checkpoints

You can download the pre-trained models from the following links:
- [CelebA](https://drive.google.com/file/d/11zaWowtEvU_rmAXnEe66x9tXzOdNbQrs/view?usp=drive_link)
- [LSUN-Bedrooms](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt)
- [ImageNet-64](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt)
- [ImageNet-256](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64_256_upsampler.pt) (64 → 256 upsampler)

Once downloaded, place the model file in a desired folder, referred to as ```[your_model_path]```. The configuration for the CelebA model is as follows:
```
--diffusion_steps 1000 --noise_schedule cosine --image_size 64 --class_cond False --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order True --learn_sigma True --dropout 0.1 --use_scale_shift_norm True --use_fp16 True
```

For configurations of LSUN-Bedrooms and ImageNet models, refer to this [link](https://github.com/openai/guided-diffusion).


## 3. Minority generation with Boost-and-Skip
In these examples, we will generate 100 samples with a batch size of 4. Feel free to adjust these values as needed.
```
SAMPLE_FLAGS="--timestep_respacing 250 --batch_size 4 --num_samples 100"
```

### 1) CelebA, LSUN-Bedrooms, and ImageNet-64
To generate samples from the CelebA model, use the following command:
```
MODEL_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --image_size 64 --class_cond False --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order True --learn_sigma True --dropout 0.1 --use_scale_shift_norm True --use_fp16 True"
python image_sample.py $MODEL_FLAGS $SAMPLE_FLAGS --model_path [your_model_path] --out_dir [your_out_dir] --gamma 4.0 --Delta_t 3
```
### 2) ImageNet-256
To replicate the generation of ImageNet-256 as described in our paper, you first need to generate samples using the ImageNet-64 model with our guidance and then upsample them to 256 x 256 using the ImageNet-256 model with ancestral sampling.

Use this command to generate samples from the ImageNet-64 model:
```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
python image_sample.py $MODEL_FLAGS $SAMPLE_FLAGS --model_path [your_model_path] --out_dir [your_out_dir] --gamma 2.5 --Delta_t 3
```
Then, upsample the generated samples to 256 x 256 with this command:
```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --large_size 256  --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python super_res_sample.py $MODEL_FLAGS $SAMPLE_FLAGS --model_path [your_model_path] --base_samples [imagenet64_npz_path] --out_dir [your_out_dir]
```

## Citation
If you find this repository useful, please cite our paper:
```
@article{um2025boost,
  title={Boost-and-Skip: A Simple Guidance-Free Diffusion for Minority Generation},
  author={Um, Soobin and Kim, Beomsu and Ye, Jong Chul},
  journal={arXiv preprint arXiv:2502.06516},
  year={2025}
}
```
