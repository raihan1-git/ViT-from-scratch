
# Vision Transformer (ViT) from Scratch in PyTorch 👁️

An educational, purely from-scratch implementation of the Vision Transformer (ViT) architecture using PyTorch. This repository demonstrates how to build, optimize, and train a ViT on the CIFAR-10 without relying on pre-built transformer libraries like Hugging Face or standard PyTorch `nn.Transformer` modules.

## 🚀 Project Highlights

* **Custom Multi-Head Self-Attention:** The MHA mechanism is written entirely from scratch to expose the underlying matrix multiplications, reshaping, and tensor permutations ($Q, K, V$ routing).

* **Modular Architecture:** Easily configurable hyperparameters to test the effects of varying `embed_dim`, `num_heads`, and transformer `depth`.

## 🧠 Architecture Overview



This model closely follows the original paper *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (Dosovitskiy et al., 2020), adapted for the 32x32 resolution of CIFAR datasets.

### Core Components Built:
1. **Patch Embedding:** Uses `nn.Conv2d` for highly efficient, simultaneous patch extraction and linear projection.
2. **Positional Embeddings & CLS Token:** Learnable parameters initialized with Truncated Normal distribution to stabilize early training.
3. **Transformer Encoder:** * Pre-Norm architecture (`LayerNorm` applied before attention and MLP).
   * Custom Scaled Dot-Product Multi-Head Attention.
   * MLP block with GELU activation and strategic Dropout regularization.

## 📊 Results & Performance

Training Vision Transformers from scratch on small datasets (without ImageNet pre-training) is notoriously difficult due to their lack of inductive bias. Through aggressive data augmentation (Random Crops, Flips) and targeted regularization, this model achieves strong baseline results.

| Dataset | Classes | Best Validation Accuracy | Note |
| :--- | :---: | :---: | :--- |
| **CIFAR-10** | 10 | **~69.0%** | Achieved using 4 blocks, 4 heads, 64 embed_dim |


## 🛠️ Usage & Installation

### Prerequisites
* Python 3.8+
* PyTorch & Torchvision
* Matplotlib & NumPy

### Running via Google Colab
The easiest way to explore the code and train the models is through the provided Google Colab notebooks:
* [🔗 Train ViT on CIFAR-10](Insert_Your_Colab_Link_Here)

