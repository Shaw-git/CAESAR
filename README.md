# CAESAR  
**A Unified Framework of Foundation and Generative Models for Efficient Compression of Scientific Data**

---

## 📖 Overview  

We introduce CAESAR, a new framework for spatio-temporal scientific data reduction that stands for Conditional AutoEncoder with Super-resolution for Augmented Reduction. The baseline model, CAESAR-V, is built on a standard variational autoencoder with scale hyperpriors and super-resolution modules to achieve high compression. It encodes data into a latent space and uses learned priors for compact, information-rich representation. The enhanced version, CAESAR-D, begins by compressing keyframes using an autoencoder and extends the architecture by incorporating conditional diffusion to interpolate the latent spaces of missing frames between keyframes. This enables high-fidelity reconstruction of intermediate data without requiring their explicit storage.
Additionally, we develop a GPU-accelerated postprocessing module that enforces error bounds on the reconstructed data, achieving real-time compression while maintaining rigorous accuracy guarantees. Combined together, this offers a set of solutions that balance compression efficiency, reconstruction accuracy, and computational cost for scientific data workflows.
Experimental results across multiple scientific datasets demonstrate that our framework achieves significantly better NRMSE rates compared to rule-based compressors such as SZ3, especially for higher compression ratios.

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
| `caesar_v.pth`           | Variational Autoencoder Based Compression       | [Download](https://yourdomain.com/models/caesar_v.pth) |
| `caesar_d.pth`           |Generative Model Based Compression               | [Download](https://yourdomain.com/models/caesar_sr.pth)|

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
