**Read this in other languages: [English](README.md), [中文](README_zh.md).**
## Welcome to star! More features are coming soon!
# Alpamayo-1-Local
VLA model, a local, offline‑running adaptation of NVIDIA’s Alpamayo. Add dataset process, visualization, fine-tuning, RL fine-tuning, consistency training.

## Visualization 

| paras             | values                                 |
|-------------------|----------------------------------------|
| clip‑id           | eed514a0‑a366‑4550‑b9bd‑4c296c531511   |
| t0‑us             | 10000000                               |

| result            | values                                 |
|-------------------|----------------------------------------|
| Chain‑of‑thought  | Adapt speed for the left curve ahead   |
| minADE            | 1.8058 m                               |

<img src="images/result_alpamayo.webp" width="70%" alt="alpamayo result">

## Usage
### ENV
```bash
python==3.12
cuda==12.1
```
```bash
git clone https://github.com/IP127000/Alpamayo-VLA-Local.git
```
```bash
cd Alpamayo-VLA-Local
```
```bash
pip install -r requirements.txt
```
```bash
python inference.py
```

### About dataset
You  needn't to download the full [dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles); you only need to download one or a few clips.
### For images inference 
Stay tuned!

# For SFT
Step 1: Split Alpamayo into two parts, Qwen-3_VL and the diffusion module, and add traj and action tokens to the Qwen-3_VL tokenizer.
Step 2: Fine-tune Qwen-3_VL.
Step 3: Fine-tune the diffusion module.

**The code is currently being released.**

# For RL
From a technical perspective, Alpamayo’s reinforcement fine-tuning is quite straightforward. Essentially, it is just the RL of Qwen-3_VL, where the VLM outputs CoT and action tokens through rollouts. This process is unrelated to the diffusion component, as described in the paper.
Stay tuned!


