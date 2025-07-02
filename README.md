# CAESAR  
**A Unified Framework of Foundation and Generative Models for Efficient Compression of Scientific Data**

---

## 📖 Overview  

CAESAR is a scalable, deep learning-based framework for efficient lossy compression of scientific data. It integrates foundation autoencoder models and generative diffusion-based models with super-resolution modules to achieve high compression ratios while preserving critical scientific structures.

This repository provides:
- Code for evaluation  
- Pretrained model weights  
- Dataset download instructions  

---

## 📦 Installation  

### 1️⃣ Clone the repository  

```bash
git clone https://github.com/yourusername/CAESAR.git
cd CAESAR
```

### 2️⃣ Install dependencies  

We recommend using Python 3.10+ and a virtual environment (e.g., `conda` or `venv`).

```bash
pip install -r requirements.txt
```

---

## 📝 Pretrained Models  

We provide 3 pretrained models for evaluation:

| Model                   | Description                                     | Download Link                                           |
|:------------------------|:------------------------------------------------|:-------------------------------------------------------|
| `caesar_v.pth`           | Variational Autoencoder-based compression       | [Download](https://yourdomain.com/models/caesar_v.pth) |
| `caesar_d.pth`           |Generative model based compression               | [Download](https://yourdomain.com/models/caesar_sr.pth)|

> 📂 Place downloaded models into the `./pretrained/` folder.

---

## 📊 Datasets  

Example scientific datasets used in this work:

| Dataset         | Description                          | Download Link                                                        |
|:----------------|:--------------------------------------|:---------------------------------------------------------------------|
| **Liangji Change it**         | Combustion ignition dataset            | [Zenodo](https://doi.org/10.5281/zenodo.6352377)                     |

Download and organize datasets into the `./data/` folder as per instructions in `data/README.md`.

---

## 🚀 Usage  

### Run compression on dataset  

```bash
python compress.py --config configs/s3d.yaml --pretrained pretrained/caesar_v.pth
```

### Run decompression  

```bash
python decompress.py --config configs/s3d.yaml --pretrained pretrained/caesar_v.pth
```

---

## 📄 Citation  

If you use CAESAR in your work, please cite:

```
@article{li2025caesar,
  title={CAESAR: A Unified Framework of Foundation and Generative Models for Efficient Compression of Scientific Data},
  author={Li, Xiao and others},
  journal={Supercomputing Conference 2025},
  year={2025}
}
```

---

## 📬 Contact  

For questions or feedback, feel free to contact **Xiao Li** at `xiaoli@ufl.edu`.

---
