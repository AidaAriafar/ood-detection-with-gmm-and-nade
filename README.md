# Out-of-Distribution Detection with GMM and NADE

Course project for **Introduction to Machine Learning** (Dr. S. Amini), Department of Electrical Engineering, Sharif University of Technology — Winter 1404.

Two density-based approaches to out-of-distribution (OOD) detection, both implemented from scratch:

- **Phase 1** — Gaussian Mixture Models trained with the EM algorithm (NumPy only), applied to OOD detection on toy 2D data and to color image segmentation.
- **Phase 2** — Neural Autoregressive Distribution Estimator (NADE) in PyTorch, trained on binarized MNIST for OOD detection and image generation.

**Authors:** Kiarash Hamidieh (402111056), Aida Ariafar (402101193)

---

## Repository contents

| File | Description |
| --- | --- |
| `Phase_1_ML_Project_402111056_402101193_.ipynb` | GMM + EM from scratch, 2D OOD detection, RGB image segmentation |
| `Phase2_MLproject_402111056_402101193.ipynb` | NADE implementation, training, OOD evaluation, sampling |
| `ML_project_402111056_402101193_.pdf` | Phase 1 report (theory questions and derivations) |
| `phase2_MLproject_402111056_402101193_.pdf` | Phase 2 report (theory + experimental results) |

---

## Phase 1 — GMM and EM

### Implementation

A `GMM` class built on NumPy, with no scikit-learn dependency:

- `gaussian_pdf` — vectorized multivariate Gaussian density with `1e-6` ridge regularization on the covariance for numerical stability.
- `e_step` — computes responsibilities `γ_ik = π_k N(x_i | μ_k, Σ_k) / Σ_j π_j N(x_i | μ_j, Σ_j)`.
- `m_step` — closed-form updates for weights, means, and covariances from the responsibilities.
- `fit` — iterates E/M steps until the mean log-likelihood changes by less than `tol`.
- `score_samples` — per-sample log-likelihood, used as the anomaly score.

### OOD detection on 2D data

In-distribution data is a two-component Gaussian mixture (600 points); OOD data is 200 points drawn uniformly over `[-6, 6]²`. After fitting a 2-component GMM to the ID data, the threshold is set at the 5th percentile of ID log-likelihoods and anything below it is flagged OOD.

| Metric | Value |
| --- | --- |
| Threshold (5th percentile of ID log-likelihood) | −4.8876 |
| True negative rate (ID correctly kept) | 95.00% |
| True positive rate (OOD correctly flagged) | 83.00% |

EM converged in 14 iterations.

### Image segmentation

The same GMM is reused for clustering-based segmentation: images are resized to 256×256, each pixel is treated as a 3-dimensional RGB sample, and the fitted component assignments are reshaped back into a label map. Run on five `skimage` sample images (Shepp-Logan phantom, retina, coffee, chelsea, rocket) with K between 3 and 4.

### Theory (see the Phase 1 report)

1. Explicit derivation of the Gaussian log-likelihood and proof that the decision region `ln p(x) ≥ τ` is an ellipsoid, with the exact boundary `(x − μ)ᵀΣ⁻¹(x − μ) = −2τ − d ln(2π) − ln|Σ|`.
2. Why the log-sum structure of a mixture has no closed-form maximizer, and derivation of the ELBO via Jensen's inequality.
3. Full M-step derivation for the means, with the weighted-average interpretation.
4. The curse of dimensionality: `E[ln p(x)] = −(d/2)ln(2πσ²) − d/2`, the typical-set argument, and why simple structure OOD samples near the mean can score a *higher* likelihood than real ID data.

---

## Phase 2 — NADE on binarized MNIST

### Setup

- **Data:** MNIST, pixels binarized at threshold 0.5, flattened to D = 784.
- **ID:** digits 0–4 (24,476 train / 6,120 val / 5,139 test). **OOD:** digits 5–9 (4,861 test), never seen during training.
- **Model:** hidden dimension H = 500; parameters `W ∈ ℝ^{H×D}`, `V ∈ ℝ^{D×H}`, biases `b`, `c`, Xavier initialization.
- **Training:** Adam, lr = 1e-3, batch size 128, gradient-norm clipping at 1.0.

### Model

The joint is factorized autoregressively, `p(x) = Π_d p(x_d | x_<d)`, and each conditional is a Bernoulli parameterized by a shared hidden layer. The key efficiency trick is the recursion

```
a_1   = c
a_d+1 = a_d + W[:, d] · x_d
```

which keeps a running pre-activation instead of recomputing `W[:, <d] x_<d` at every step, dropping the cost per image from O(HD²) to O(HD). Training minimizes the negative log-likelihood, which for binary pixels is exactly a summed binary cross-entropy — computed here with `binary_cross_entropy_with_logits` for numerical stability.

### Results

Training and validation NLL both decrease steadily and stay close together, so the model is learning digit structure rather than memorizing.

Using log-likelihood as the anomaly score (label OOD = 1, score = NLL):

| Split | Average log-likelihood |
| --- | --- |
| ID test (0–4) | −71.70 |
| OOD test (5–9) | −101.65 |

**AUROC: 0.7959** (reported in `phase2_MLproject_402111056_402101193_.pdf`, 10 epochs). Re-running the notebook for 20 epochs gives −66.75 / −96.22 and AUROC 0.7997.

The histograms of ID and OOD log-likelihoods are clearly shifted apart, with some residual overlap — expected given that 5–9 share stroke statistics with 0–4.

### Generation

Sampling proceeds pixel by pixel: compute `x̂_d = σ(V_d h_d + b_d)`, draw `x_d ~ Bernoulli(x̂_d)`, update the accumulator, repeat. Because each pixel depends on the previous ones, sampling is inherently sequential across all 784 dimensions and cannot be batched along D the way scoring can. Generated samples show recognizable digit-like structure from the 0–4 classes, with the speckle noise typical of per-pixel Bernoulli sampling.

---

## Running the code

```bash
pip install numpy matplotlib scikit-learn scikit-image opencv-python torch torchvision tqdm
```

Then open either notebook and run the cells top to bottom. Phase 2 downloads MNIST automatically into `./data` and saves the trained model to `nade_mnist.pth`. A GPU is recommended for Phase 2 — the autoregressive forward pass loops over all 784 pixels.

---

## References

- Bishop, *Pattern Recognition and Machine Learning*, Ch. 9 (EM for Gaussian mixtures).
- Larochelle & Murray, *The Neural Autoregressive Distribution Estimator*, AISTATS 2011.
- Nalisnick et al., *Do Deep Generative Models Know What They Don't Know?*, ICLR 2019 — on the typical-set failure mode discussed in Phase 1.
