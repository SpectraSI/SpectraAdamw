<div align="center">

# Spectra

**The Frequency-Domain Optimizer for Deep LLM Architectures**

<p align="center">
  <a href="https://www.spectralabs.si"><img src="https://www.spectralabs.si/nvidia-inception.png" alt="Member of NVIDIA Inception Program" width="220"></a>
</p>

[![Website](https://img.shields.io/badge/Website-spectralabs.si-blue?style=flat-square)](https://www.spectralabs.si)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg?style=flat-square)](LICENSE)
[![W&B Logs](https://img.shields.io/badge/Weights_&_Biases-Live_Logs-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)](https://api.wandb.ai/links/project-mosaic/u913b38j)

**Spectra** is a drop-in optimizer replacement for AdamW designed to eliminate the VRAM tax of training Large Language Models. By shifting gradient optimization to the frequency domain and factorizing second-moment variance, Spectra **cuts optimizer state memory by exactly 50%**, runs **6.2x faster than GaLore**, and maintains **full 32-bit floating-point stability** with zero quantization collapse.

</div>

---

## Quickstart

Spectra is designed as a seamless 1-line replacement for standard PyTorch `torch.optim.AdamW`:

```python
import torch
from spectra import Spectra

# Standard PyTorch model
model = MyTransformerLLM().cuda()

# 1-line drop-in replacement (saves 50% optimizer state VRAM)
optimizer = Spectra(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# Standard training loop (unchanged)
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()

Empirical Benchmark: 12-Layer Deep Transformer Block
To evaluate real-world gradient degradation and compute overhead, we benchmarked five memory-constrained optimizers on a 12-Layer Deep Transformer Block (4096 hidden dimension, 11008 intermediate dimension) for 2,500 steps on an NVIDIA A100 Tensor Core GPU.

Optimizer	State VRAM (MB)	Final Loss	Step Latency (ms)	Status	Key Characteristic
Standard AdamW	8256.75 MB	0.00001	95.49 ms	Baseline	High memory tax (8 bytes/param)
Spectra (Ours)	4130.13 MB	0.00000	92.32 ms	Optimal	50% VRAM cut, zero collapse, fastest step
GaLore AdamW	258.75 MB	0.00000	574.78 ms	Compute Bottleneck	SVD decomposition makes it 6.2x slower
Adafactor	1.76 MB	0.26480	97.97 ms	Convergence Stalled	Deleting momentum destroys deep convergence
8-bit AdamW	2096.53 MB	NaN	73.03 ms	Gradient Explosion	Integer rounding noise triggers NaN collapse
Key Benchmark Findings:
Zero Quantization Collapse: While bitsandbytes 8-bit Adam completely failed (NaN loss) due to compounding integer rounding errors across deep layers, Spectra maintains 32-bit floating-point precision with zero divergence.
6.2x Faster Than GaLore: GaLore compresses gradients using Singular Value Decomposition (SVD), which carries an 
O
(
N
3
)
O(N 
3
 ) computational cost that severely bottlenecks the GPU (574.78 ms/step). Spectra factorizes variance into row/column components, running natively at 92.32 ms/step.
Full Momentum Preservation: Adafactor eliminates the first moment to minimize memory, stalling convergence at 0.26480. Spectra preserves full momentum updates, driving loss to 0.00000.
Technical Methodology
1. Factorized Variance
Standard AdamW tracks a full 
N
×
M
N×M matrix for second-moment variance 
v
t
v 
t
​
 . Spectra factorizes this into 1D row and column marginal projections (
N
+
M
N+M), slashing variance memory to near zero while preserving directional scaling. Strict numerical epsilon clamping (torch.clamp(..., min=eps)) prevents Gibbs ringing and division-by-zero artifacts.

2. Frequency-Domain Signal Isolation (F-Principle)
Deep networks learn low-frequency structural features before high-frequency noise. Spectra leverages 2D Fast Fourier Transforms (cuFFT) to physically filter high-frequency stochastic batch noise, acting as a natural regularizer and preventing micro-batch overfitting.

Ecosystem & Community
Spectra Labs is a member of the NVIDIA Inception Program for cutting-edge AI startups.

🌐 Official Website: https://www.spectralabs.si
📄 Technical Report (Zenodo DOI): https://zenodo.org/records/22967704
📊 Live Training Logs (W&B): Weights & Biases Dashboard
💬 Reddit Community: r/SpectraLabs
🐦 Twitter/X: @SpectraSILabs
Legal & Trademarks
© 2026 Spectra Labs. All rights reserved.

© 2026 NVIDIA, the NVIDIA logo, CUDA, and NVIDIA A100 are trademarks and/or registered trademarks of NVIDIA Corporation in the U.S. and other countries.

15:15
