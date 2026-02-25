**Read this in other languages: [English](README.md), [中文](README_zh.md).**
## 欢迎star！更多功能等待释放
# Alpamayo-1-Local
VLA模型，英伟达Alpamayo 1的本地、离线运行版本。添加可视化，微调，强化微调，一致性训练。

## 可视化

| 参数名            | 数值                                   |
|-------------------|----------------------------------------|
| clip‑id           | eed514a0‑a366‑4550‑b9bd‑4c296c531511   |
| t0‑us             | 10000000                               |

| 结果              | 数值                                   |
|-------------------|----------------------------------------|
| cot               | Adapt speed for the left curve ahead   |
| minADE            | 1.8058 m                               |

<img src="images/result_alpamayo.webp" width="70%" alt="alpamayo 结果">

## 使用方法
环境要求
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
