# CAESAR  
**A Unified Framework of Foundation and Generative Models for Efficient Compression of Scientific Data**

---

## 📖 Overview  

![CAESAR Framework Overview](/figures/caesar_overview.png)

We introduce **CAESAR**, a new framework for spatio-temporal scientific data reduction that stands for *Conditional AutoEncoder with Super-resolution for Augmented Reduction*. The baseline model, **CAESAR-V**, is built on a standard variational autoencoder with scale hyperpriors and super-resolution modules to achieve high compression. It encodes data into a latent space and uses learned priors for compact, information-rich representation.  

The enhanced version, **CAESAR-D**, begins by compressing keyframes using an autoencoder and extends the architecture by incorporating conditional diffusion to interpolate the latent spaces of missing frames between keyframes. This enables high-fidelity reconstruction of intermediate data without requiring their explicit storage.

Additionally, we develop a **GPU-accelerated postprocessing module** that enforces error bounds on the reconstructed data, achieving real-time compression while maintaining rigorous accuracy guarantees. Combined together, this offers a set of solutions that balance compression efficiency, reconstruction accuracy, and computational cost for scientific data workflows.

**Experimental results** across multiple scientific datasets demonstrate that our framework achieves significantly better NRMSE rates compared to rule-based compressors such as **SZ3**, especially for higher compression ratios.


## 📦 Installation  

### 1️⃣ Clone the repository  

```bash
git clone https://github.com/Shaw-git/CAESAR.git
cd CAESAR
```

### 2️⃣ Install dependencies  

We recommend using Python 3.10+ and a virtual environment (e.g., `conda` or `venv`).

```bash
pip install -r requirements.txt
```

---

### ✅ Tested Hardware
This project has been tested on:

- NVIDIA A100 80GB

- NVIDIA RTX 2080 24GB


## 📝 Pretrained Models  

We provide 4 pretrained models for evaluation:

| Model                   | Description                                     | Download Link                                           |
|:------------------------|:------------------------------------------------|:-------------------------------------------------------|
| `caesar_v.pth`                  |CAESAE-V                                         | [Google Drive](https://drive.google.com/file/d/1sVmxgdg0EdyRK2PhihVamToR2gdRu1nz/view?usp=sharing) |
| `caesar_d.pth`                  |CAESAE-D                                         | [Google Drive](https://drive.google.com/file/d/16J4Uv0RPGPAZLHqm2MlpX9SS1dr-gJW2/view?usp=sharing)|
| `caesar_v_tuning_Turb-Rot.pth`  |CAESAE-V Finetuned on Turb-Rot dataset    | [Google Drive](https://drive.google.com/file/d/1fF8MTTWofyq2ihrc1dn0yfrLtZyE1bR9/view?usp=drive_link)|
| `caesar_d_tuning_Turb-Rot.pth`  |CAESAE-D Finetuned on Turb-Rot dataset    | [Google Drive](https://drive.google.com/file/d/1EjyD93FPgwpPDbdWbW9vT1_Cph_JqTis/view?usp=drive_link)|
> 📂 Place downloaded models into the `./pretrained/` folder.

---

## 📊 Datasets  

Example scientific datasets used in this work:

| Dataset         | Description                          | Download Link                                                        |
|:----------------|:--------------------------------------|:---------------------------------------------------------------------|
| **Turb-Rot**         | Rotating turbulence dataset            | [Google Drive](https://drive.google.com/file/d/16nusjTsjxjpvRNBzusrpT2mUKi0Cg3VS/view?usp=drive_link)                     |

Download and organize datasets into the `./data/` folder as per instructions in `data/README.md`.

---

## 🗂️ Data Organization

All datasets used in this work are stored in NumPy `.npz` format and follow a standardized 5D tensor structure:
[variable, n_samples, T, H, W]
- **variable**: number of physical quantities 
- **n_samples**: number of independent spatial samples
- **T**: number of time steps per sample
- **H/W**: spatial resolution (height × width)

## 🚀 Usage  

### Run compression on dataset  

```bash
see eval_caesar.ipynb
```

---

## 📄 Citation  

If you use CAESAR in your work, please cite:

```
@inproceedings{li2025foundation,
  title={Foundation Model for Lossy Compression of Spatiotemporal Scientific Data},
  author={Li, Xiao and Lee, Jaemoon and Rangarajan, Anand and Ranka, Sanjay},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={368--380},
  year={2025},
  organization={Springer}
}

@article{li2025generative,
  title={Generative Latent Diffusion for Efficient Spatiotemporal Data Reduction},
  author={Li, Xiao and Zhu, Liangji and Rangarajan, Anand and Ranka, Sanjay},
  journal={arXiv preprint arXiv:2507.02129},
  year={2025}
}

```

---

## 📬 Contact  

For questions or feedback, feel free to contact **Xiao Li** at `xiao.li@ufl.edu`.

---
