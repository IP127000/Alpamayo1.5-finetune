
# 🌟 Alpamayo-1&1.5
[![GitHub stars](https://img.shields.io/github/stars/IP127000/Alpamayo-VLA-Local?style=social)](https://github.com/IP127000/Alpamayo-VLA-Local/stargazers)

**Read this in other languages:** [English](README.md) | [中文](README_zh.md)

Offline adaptation of NVIDIA’s Alpamayo with dataset processing, visualization, fine-tuning, RL fine-tuning, and consistency training.

---

## 📌 Table of Contents
1. [Visualization](#-visualization)
2. [Usage](#-usage)
3. [Dataset](#-about-dataset)
4. [SFT](#-for-sft-supervised-fine-tuning)
5. [RL](#-for-rl-reinforcement-learning)
6. [Contributing](#-contributing)

---

## 🔹 Visualization 

**Parameters:**

| Parameter | Value |
|-----------|-------|
| clip‑id   | eed514a0‑a366‑4550‑b9bd‑4c296c531511 |
| t0‑us     | 10000000 |

**Results:**

| Metric             | Value |
|-------------------|-------|
| Chain-of-thought  | Adapt speed for the left curve ahead |
| minADE            | 1.8058 m |

<p align="center">
  <img src="images/result_alpamayo.webp" width="70%" alt="Alpamayo result">
</p>

---

## ⚙️ Usage

### 1️⃣ Environment
```bash
python==3.12
cuda==12.1
````

### 2️⃣ Clone repository

```bash
git clone https://github.com/IP127000/Alpamayo-VLA-Local.git
cd Alpamayo-VLA-Local
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run inference

```bash
python inference.py
```

---

## 🗂 About dataset

You don't need the full [dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles);
only download one or a few clips for testing.

### 📸 Image inference

Stay tuned for updates!

---

## 🛠 For SFT (Supervised Fine-Tuning)

1. Split Alpamayo into two parts: **Qwen-3_VL** + **diffusion module**.
   Add trajectory & action tokens to Qwen-3_VL tokenizer.
2. Fine-tune **Qwen-3_VL**.
3. Fine-tune the **diffusion module**.

> **Note:** Code release in progress.

---

## 🎯 For RL (Reinforcement Learning)

RL fine-tuning of Alpamayo focuses on **Qwen-3_VL**:

* VLM outputs **CoT** + **action tokens** via rollouts.
* Diffusion module is unaffected.

Stay tuned!

---

## 🤝 Contributing

Feel free to **star⭐, fork🍴, and submit PRs** to help improve Alpamayo-1-Local!

---
