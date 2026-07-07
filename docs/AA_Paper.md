# Towards Sparse Adversarial Perturbations: A Gradient-Guided Approach for Efficient Image Attacks

**Tác giả:** Xuan-Cuong Nguyen, Nhat-Quang Truong

> ### Abstract
> 
> 
> Adversarial attacks expose the vulnerability of deep neural networks by introducing carefully crafted perturbations to input images. While iterative gradient-based methods achieve high attack success rates, they typically rely on dense perturbations affecting most pixels. This raises a fundamental question: are all pixels equally important for adversarial manipulation?
> In this work, we investigate the role of pixel-level importance in adversarial attacks and propose a gradient-guided approach that prioritizes perturbations on the most influential regions of the input. By leveraging gradient magnitude as a proxy for importance, our method selectively updates a subset of pixels during optimization. Experimental results show that strong attacks can be achieved while modifying only a small fraction of pixels, leading to improved perceptual quality and more interpretable perturbations. Our findings suggest that adversarial vulnerability is inherently sparse, opening new directions for efficient and structured attacks.

---

## 1. Introduction

Deep neural networks have demonstrated remarkable performance in image classification tasks, yet they remain highly vulnerable to adversarial perturbations. Standard gradient-based attack methods, such as the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD), generate adversarial examples by maximizing the classification loss with respect to the input. Despite their effectiveness, these methods share a common property: the generated perturbations are characteristically dense, affecting nearly all pixels within the image. This pervasive modification strategy raises several fundamental questions regarding the necessity of dense perturbations, specifically whether all pixels are equally important for adversarial attacks, what mathematically defines an "important" pixel during optimization, and whether successful attacks can be constructed by modifying only a minute subset of pixels.

Our motivation stems from empirical observations indicating that adversarial vulnerability is spatially non-uniform and potentially sparse. Experiments demonstrate that high attack success rates can be achieved even when perturbations are restricted to a fractional subset of the input space. Furthermore, we observe that the gradient magnitude varies significantly across different spatial regions of an image, and visual similarity is substantially improved when perturbations are localized rather than globally distributed. Collectively, these observations suggest that by identifying and exploiting the most influential pixels, we can design adversarial attacks that are not only computationally efficient but also highly imperceptible.

---

## 2. Related Work

### 2.1 Gradient-based Attacks

Gradient-based adversarial attacks have established the foundation for evaluating model robustness. These approaches generally operate by optimizing the classification loss with respect to the continuous input space, leveraging first-order gradient information to maximize the network's prediction error.

* **Fast Gradient Sign Method (FGSM):** Introduced by Goodfellow et al., FGSM is a foundational, single-step attack that computes the gradient of the loss function with respect to the input image. By applying a perturbation scaled by a magnitude $\epsilon$ strictly in the direction of the gradient's sign, it quickly generates adversarial examples. While computationally highly efficient, its single-step nature often results in suboptimal perturbations that fail to bypass more robust defenses, and it inherently alters nearly all pixels in the image.
* **Basic Iterative Method (BIM):** To overcome the limitations of single-step optimization, Kurakin et al. proposed BIM, which effectively applies the FGSM strategy iteratively. At each step, a smaller perturbation step size is utilized, and the intermediate image is clipped to ensure it remains within the valid $L_\infty$ $\epsilon$-neighborhood of the original input. This iterative refinement significantly increases the attack success rate but continues to produce pervasive, dense modifications across the entire spatial domain.
* **Projected Gradient Descent (PGD):** Recognized as the universal first-order adversary, PGD formalizes the iterative attack process by incorporating random uniform initializations within the $\epsilon$-ball prior to optimization. Madry et al. demonstrated that PGD represents the strongest local first-order attack, making it the gold standard for evaluating adversarial robustness. However, like its predecessors, PGD optimizes the continuous input space globally, inevitably producing dense, visually degrading adversarial noise without spatial interpretability.

### 2.2 Sparse Attacks

In contrast to dense perturbation methods, sparse attacks aim to deceive neural networks by altering only a minimal fraction of the input features, a problem typically formulated under an $L_0$ norm constraint.

* **Classical Sparse Approaches (JSMA & One-pixel):** Shifting focus towards extreme sparsity, Papernot et al. introduced JSMA. This method constructs a saliency map by computing the forward derivative (Jacobian matrix) of the network, explicitly identifying which specific pixels exert the highest influence on the target class probabilities. While highly effective at minimizing the $L_0$ norm, the repeated computation of the full Jacobian matrix renders JSMA computationally prohibitive. Pushing the boundaries of spatial sparsity, Su et al. demonstrated that deep neural networks can be deceived by altering just a single pixel using Differential Evolution, though this requires extensive model queries.
* **Modern White-Box Sparse Attacks:** Recent advancements have focused on more efficient optimization-based sparse attacks. **SparseFool** (Modas et al.) extends the DeepFool geometry to find sparse perturbations by iteratively projecting onto the decision boundary. **GreedyFool** (Dong et al.) employs a two-stage approach using gradient saliency and a distortion-aware penalty to greedily drop less important pixels. Other notable methods include **SAIF** and **AutoAdversary**, which frame sparse attacks through the lens of reinforcement learning and specialized search strategies, and **Homotopy-based methods**, which trace solution paths for sparse imperceptible attacks.
* **Closest State-of-the-Art (Sparse-PGD):** The most direct precursor to our approach is **Sparse-PGD** (Zhong & Liu), which integrates $L_0$ projection directly into the PGD optimization framework. While Sparse-PGD relies primarily on coordinate-wise magnitude thresholding, our gradient-guided approach extends this concept by incorporating dynamic masking and spatial scoring mechanisms to better preserve perceptual quality and focus on semantic regions.

---

## 3. Problem Formulation

We consider a standard image classification setting where a trained deep neural network is denoted as $f(\cdot)$. Let $x \in \mathbb{R}^n$ represent an original, clean input image consisting of $n$ total pixels, and let $y$ denote its corresponding ground-truth label. The fundamental objective of a standard adversarial attack is to find a continuous perturbation $\delta$ (of the same dimension as $x$) that maximizes the classification loss $\mathcal{L}$, thereby inducing a misclassification by the model. Formally, this adversarial optimization problem is defined as:

$$\max_{\delta} \ \mathcal{L}(f(x + \delta), y)$$

To ensure that the generated adversarial example remains visually imperceptible and indistinguishable from the original image to the human eye, it is standard practice to bound the maximum magnitude of $\delta$. This is typically achieved by enforcing an $L_\infty$ norm constraint, which restricts the maximum deviation of any single pixel channel to a predefined, small perturbation threshold $\epsilon$:

$$\|\delta\|_\infty \leq \epsilon$$

Traditional gradient-based methods primarily focus on satisfying this $L_\infty$ constraint, which inevitably leads to dense perturbations where almost every pixel in the image is modified up to the allowed limit $\epsilon$. In contrast to this conventional approach, our work introduces an extended question: *Can we strategically restrict the perturbations to only a small, highly influential subset of pixels while still maintaining the overall strength and success rate of the attack?* To formalize this targeted objective, we introduce an implicit sparsity constraint into the adversarial optimization process. By limiting the $L_0$ norm of the perturbation—which counts the total number of non-zero elements (i.e., the strictly modified pixels)—we ensure that the number of altered pixels is significantly smaller than the total number of pixels $n$:

$$\|\delta\|_0 \ll n$$

By unifying the continuous magnitude bound with this spatial sparsity constraint, our formulation shifts the adversarial attack from a global noise addition process to a highly localized, feature-selective optimization problem.

---

## 4. Methodology

### 4.1 Sparsity-Constrained Adversarial Optimization

Finding an exact solution for an $L_0$-constrained adversarial perturbation is notoriously NP-hard due to its combinatorial nature. Standard first-order methods relax this spatial constraint entirely, yielding dense perturbations. To formally motivate our approach, consider the first-order Taylor expansion of the adversarial objective around the current input $x$:

$$\mathcal{L}(f(x + \delta), y) \approx \mathcal{L}(f(x), y) + \langle \nabla_x \mathcal{L}(f(x), y), \delta \rangle$$

To maximize this objective under the joint constraints $\|\delta\|_\infty \leq \epsilon$ and $\|\delta\|_0 \leq k$, the optimal greedy strategy allocates the perturbation budget exclusively to the spatial coordinates $i$ that maximize the gradient magnitude $|\nabla_{x_i} \mathcal{L}|$. This theoretical alignment naturally dictates the use of the gradient magnitude as an optimal proxy for evaluating pixel-level adversarial vulnerability.

### 4.2 Gradient-Guided Mask Generation

We formulate the spatial feature selection process by extracting an importance map $S \in \mathbb{R}^n$ at the optimization step $t$. Let the gradient vector be defined as:

$$g_t = \nabla_x \mathcal{L}(f(x_t), y)$$

The importance map is given by the element-wise absolute magnitude of this gradient:

$$S_t = |g_t|$$

To strictly enforce the $L_0$ budget, we compute the perturbation $\delta_t = x_t - x$ and utilize a projection operator $\Pi_0$ that retains only the top-$k$ spatial pixels with the largest accumulated perturbation magnitude. This prevents the "notation drift" problem where dynamic mask changes over iterations unintentionally bloat the final $L_0$ norm beyond the specified budget.

### 4.3 Algorithm: Top-$k$ Gradient-Guided PGD

The complete attack procedure is outlined in Algorithm 1.

```text
Algorithm 1: Gradient-Guided Sparse Attack (Top-k PGD)
Input: Image x, label y, model f, iterations T, step size α, budget k, bound ε
Output: Adversarial image x_adv

1: Initialize δ_0 = 0
2: for t = 0 to T - 1 do
3:     x_t = x + δ_t
4:     g_t = ∇_x L(f(x_t), y)
5:     
6:     # Compute importance score and mask
7:     S_t = |g_t|
8:     M_t = TopK_Mask(S_t, k)
9:     
10:    # Gradient update
11:    δ_{t+1} = δ_t + α * sign(g_t) * M_t
12:    
13:    # Magnitude constraint
14:    δ_{t+1} = clamp(δ_{t+1}, -ε, ε)
15:    
16:    # Strict spatial L0 projection
17:    δ_{t+1} = Project_L0(δ_{t+1}, k)
18:    
19:    # Image bounds
20:    δ_{t+1} = clamp(x + δ_{t+1}, 0, 1) - x
21: end for
22: return x + δ_T
```

### 4.4 Interpretation: Attacks as Spatial Feature Selection

This formulation introduces a fundamental paradigm shift in the conceptualization of adversarial generation. Traditional first-order adversaries, such as standard PGD, optimize over the entire unconstrained image manifold, treating the input space as a uniform canvas for noise addition. In contrast, our approach reframes adversarial attacks as a constrained submodular optimization problem over the input features. By confining the perturbation $\delta$ to a lower-dimensional subspace spanned by the most salient pixels, we empirically demonstrate that successful adversarial deception does not require pervasive noise. Instead, network fragility is heavily concentrated in highly localized semantic regions, allowing us to disentangle true adversarial vulnerability from dense pixel manipulation.

---

## 5. Experiments

### 5.1 Experimental Setup

We evaluate the proposed method on the CIFAR-10 dataset, which consists of 60,000 images across 10 classes. Two types of models are considered in our experiments:

1. **Standard model:** Trained on CIFAR-10 using conventional empirical risk minimization.
2. **Robust model:** Obtained via adversarial training on CIFAR-10, designed to improve resistance against gradient-based attacks.

All experiments are conducted on a subset of 1,000 test images to ensure consistency and fair comparison across different attack methods. We compare our approach with several widely used gradient-based attacks, including FGSM, BIM, and PGD, as well as modern sparse baselines including Sparse-PGD, SparseFool, and GreedyFool.

To comprehensively evaluate attack performance, we report multiple metrics, including Attack Success Rate (ASR), classification accuracy, sparsity (i.e., the proportion of modified pixels), Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR), and Learned Perceptual Image Patch Similarity (LPIPS).

### 5.2 Evaluation Metrics

Let $x$ denote the clean input image, $x_{adv} = x + \delta$ the generated adversarial example, $y$ the ground-truth label, $f(\cdot)$ the target classifier, and $N$ the total number of test samples.

* **Accuracy (Acc):** Measures the overall classification accuracy of the target model on the generated adversarial examples. A stronger attack results in lower accuracy.

$$\text{Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(f(x_{adv}^{(i)}) = y^{(i)})$$


* **Attack Success Rate (ASR):** Quantifies the proportion of originally correctly classified images that are successfully misclassified by the model after the attack.

$$\text{ASR} = \frac{\sum_{i=1}^{N} \mathbb{I}(f(x^{(i)}) = y^{(i)} \land f(x_{adv}^{(i)}) \neq y^{(i)})}{\sum_{i=1}^{N} \mathbb{I}(f(x^{(i)}) = y^{(i)})} \times 100\%$$


* **$L_0$ Norm and Sparsity (%):** The $L_0$ norm evaluates the strict number of modified pixels in the image. Spatial sparsity is the percentage of pixels that remain completely unaltered compared to the total number of pixels $n$.

$$\|\delta\|_0 = \sum_{j=1}^{n} \mathbb{I}(\delta_j \neq 0), \quad \text{Sparsity} = \left( 1 - \frac{\|\delta\|_0}{n} \right) \times 100\%$$


* **$L_2$ and $L_\infty$ Norms:** Quantify the magnitude of the adversarial perturbation.

$$\|\delta\|_2 = \sqrt{\sum_{j=1}^{n} \delta_j^2}, \quad \|\delta\|_\infty = \max_{j} |\delta_j|$$


* **Structural Similarity Index Measure (SSIM):** Evaluates the perceptual visual similarity between the clean image $x$ and the adversarial image $x_{adv}$ by considering changes in luminance, contrast, and structure.

$$\text{SSIM}(x, x_{adv}) = \frac{(2\mu_x \mu_{x_{adv}} + c_1)(2\sigma_{x, x_{adv}} + c_2)}{(\mu_x^2 + \mu_{x_{adv}}^2 + c_1)(\sigma_x^2 + \sigma_{x_{adv}}^2 + c_2)}$$


* **Peak Signal-to-Noise Ratio (PSNR):** Measures the ratio between the maximum possible pixel value ($\text{MAX}_I$) and the power of the perturbation noise, evaluated via the Mean Squared Error (MSE).

$$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}(x, x_{adv})} \right)$$

* **Learned Perceptual Image Patch Similarity (LPIPS):** Evaluates perceptual distance using deep features, offering a metric more aligned with human perception than SSIM or PSNR.


### 5.3 Results on Standard Model

**Table 1:** Comparison of attack performance on the standard CIFAR-10 model. Sparse attacks achieve competitive attack success rates while modifying significantly fewer pixels and preserving higher perceptual quality.

| Attack | K | Iter | Acc (%) | ASR (%) | $L_0$ | Sparsity (%) | $L_2$ | $L_\infty$ | SSIM | PSNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clean | - | 0 | 94.8 | 0.0 | 0 | 100.0 | 0.000 | 0.000 | 1.0000 | $\infty$ |
| FGSM | - | 1 | 30.5 | 68.0 | 1021 | 0.3 | 1.724 | 0.031 | 0.9380 | 30.14 |
| BIM | - | 10 | 0.0 | 100.0 | 997 | 2.6 | 1.209 | 0.031 | 0.9692 | 33.22 |
| PGD | - | 10 | 0.0 | 100.0 | 1023 | 0.1 | 1.246 | 0.031 | 0.9666 | 32.96 |
| SparseFool | - | 20 | 5.2 | 94.5 | 312 | 69.5 | 0.810 | 1.000 | 0.9850 | 34.10 |
| Sparse-PGD | 0.1 | 10 | 7.5 | 92.0 | 102 | 90.0 | 0.315 | 0.031 | 0.9960 | 45.10 |
| GreedyFool | 0.1 | 10 | 6.8 | 92.8 | 102 | 90.0 | 0.310 | 0.031 | 0.9965 | 45.30 |
| **Ours (Top-k PGD)** | 0.1 | 10 | 5.9 | 93.7 | 102 | 90.0 | 0.320 | 0.031 | **0.9968** | **45.50** |
| **Ours (Top-k PGD)** | 0.5 | 10 | 0.2 | 99.8 | 512 | 50.0 | 0.849 | 0.031 | 0.9818 | 36.28 |
| **Ours (Top-k PGD)** | 1.0 | 10 | 0.0 | 100.0 | 1020 | 0.4 | 1.247 | 0.031 | 0.9664 | 32.96 |


Dense attacks such as PGD and BIM achieve near-perfect attack success rates (close to 100%), but at the cost of modifying almost all pixels in the input image. In contrast, the proposed method achieves comparable attack performance while modifying only a subset of pixels. Notably, even with a significantly reduced number of perturbed pixels (e.g., $k=0.1$ modifying only 102 pixels), the attack remains highly effective and competitive with modern baselines like Sparse-PGD and GreedyFool.

Furthermore, as the number of selected pixels decreases, the perceptual quality of the adversarial examples improves substantially. This is reflected by higher SSIM values (approaching 0.99) and increased PSNR compared to dense attacks. These results indicate that effective adversarial perturbations do not require dense modifications across the entire image, and can instead be concentrated on a limited set of important pixels.

### 5.4 Attack Progression Analysis

*(Xem chi tiết tài sản ảnh tại các file tương ứng: `progression_asr_standard.png`, `progression_accuracy_standard.png`, `progression_ssim_standard.png`, `progression_psnr_standard.png`)*

Figure 1 illustrates how attack performance evolves over iterations. Dense attacks such as PGD and BIM rapidly achieve near-perfect ASR within the first few iterations, leading to a sharp drop in model accuracy. In contrast, sparse attacks exhibit a more gradual increase in ASR, particularly under strong sparsity constraints (e.g., $k=0.1$), indicating slower but stable convergence.

In terms of perceptual quality, sparse attacks consistently maintain higher SSIM and PSNR throughout the optimization process. Even at higher sparsity levels, the visual quality remains significantly better than dense attacks, demonstrating that perturbations are more localized and less perceptible.

### 5.5 Impact of Pixel Selection (K-Ratio)

*(Xem chi tiết đồ thị thanh tại các file tương ứng: `bar_kratio_asr.png`, `bar_kratio_accuracy.png`, `bar_kratio_ssim.png`, `bar_kratio_psnr.png`)*

Figure 2 summarizes the impact of the number of selected pixels on attack performance. As the K-ratio increases, the attack success rate improves significantly, exhibiting a diminishing-return trend. Notably, even with a small fraction of selected pixels (e.g., $k=0.1$), the attack can still achieve high effectiveness given sufficient iterations.

At the same time, increasing K leads to a reduction in perceptual quality, as reflected by decreasing SSIM and PSNR. However, even at relatively high K values, sparse attacks maintain strong visual fidelity compared to dense methods, highlighting a favorable trade-off between attack strength and imperceptibility.

---

### 5.6 Results on Robust Model

**Table 2:** Attack performance on the adversarially trained (robust) CIFAR-10 model.

| Attack | K | Iter | Acc (%) | ASR (%) | $L_0$ | Sparsity (%) | $L_2$ | $L_\infty$ | SSIM | PSNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clean | - | 0 | 85.9 | 0.0 | 0 | 100.0 | 0.000 | 0.000 | 1.0000 | $\infty$ |
| FGSM | - | 1 | 63.0 | 26.7 | 1022 | 0.2 | 1.730 | 0.031 | 0.9621 | 30.11 |
| BIM | - | 10 | 59.0 | 31.4 | 1022 | 0.2 | 1.668 | 0.031 | 0.9664 | 30.43 |
| PGD | - | 10 | 59.5 | 30.8 | 1022 | 0.2 | 1.653 | 0.031 | 0.9665 | 30.51 |
| SparseFool | - | 20 | 78.1 | 9.0 | 451 | 55.9 | 0.920 | 1.000 | 0.9810 | 33.20 |
| Sparse-PGD | 0.1 | 10 | 75.2 | 12.4 | 102 | 90.0 | 0.331 | 0.031 | 0.9975 | 45.00 |
| GreedyFool | 0.1 | 10 | 74.0 | 13.8 | 102 | 90.0 | 0.335 | 0.031 | 0.9972 | 44.90 |
| **Ours (Top-k PGD)** | 0.1 | 10 | 73.1 | 14.9 | 102 | 90.0 | 0.342 | 0.031 | 0.9973 | 44.75 |
| **Ours (Top-k PGD)** | 0.5 | 10 | 62.6 | 27.2 | 512 | 50.0 | 1.082 | 0.031 | 0.9871 | 34.19 |
| **Ours (Top-k PGD)** | 1.0 | 10 | 59.3 | 31.0 | 1017 | 0.7 | 1.653 | 0.031 | 0.9665 | 30.51 |


*(Xem chi tiết tài sản ảnh tại các file tương ứng: `progression_asr_robust.png`, `progression_accuracy_robust.png`, `progression_ssim_robust.png`, `progression_psnr_robust.png`)*

As shown in Table 2 and Figure 3, all attack methods experience a significant drop in effectiveness compared to the standard model. Dense attacks such as PGD achieve only moderate ASR, reflecting the improved robustness of the model.

Sparse attacks also exhibit reduced effectiveness under strict sparsity constraints, but remain competitive overall. Notably, the attack progression becomes significantly smoother, and metrics such as SSIM and PSNR remain more stable across iterations. This behavior indicates that adversarial training constrains the optimization process and reduces the ability of gradient-based methods to exploit vulnerable directions.

### 5.7 Summary of Findings

Overall, the experimental results demonstrate that adversarial perturbations do not need to be dense to be effective. By selectively focusing on a subset of important pixels, it is possible to achieve strong attack performance while significantly improving perceptual quality. Moreover, the observed trade-off between sparsity and attack strength provides deeper insight into the structure of adversarial vulnerabilities in deep neural networks.

---

## 6. Contributions

This work makes several key contributions to the field of adversarial machine learning:

* We introduce a novel perspective on adversarial vulnerability by fundamentally demonstrating that highly effective attacks do not equate to, nor require, dense global perturbations.
* We propose a new framework centered on a gradient-guided selective perturbation mechanism, which strategically isolates and targets only the most influential input features.
* We provide robust empirical proof through rigorous, extensive experiments, confirming that these localized, sparse perturbations maintain highly competitive attack success rates compared to traditional dense methods, specifically matching modern baselines like Sparse-PGD and GreedyFool.
* Our in-depth analysis explicitly evaluates the intricate trade-off between spatial sparsity and overall attack strength, offering deeper insights into the underlying optimization dynamics and model fragility.

---

## 7. Limitations

Despite the promising empirical results, this study presents certain limitations that warrant future investigation:

1. While our gradient-magnitude approach proves highly effective as a proxy for feature importance, the mathematically optimal pixel selection strategy remains undetermined, necessitating further theoretical exploration into exact submodular optimization bounds.
2. The current framework operates strictly within a white-box setting, as it relies entirely on full access to the target model's internal gradient information to construct the sparsity mask.
3. The proposed method currently treats input dimensions independently and has not yet been extended to encompass structured or contiguous sparsity (such as block-wise or region-based selections), which limits its immediate translation to physically realizable patch attacks.

---

## 8. Future Work

One promising direction for future research is to further investigate adaptive sparsity mechanisms in adversarial attacks. In our current formulation, the number of perturbed pixels is fixed throughout the optimization process. However, empirical observations from our experiments suggest that the set of influential pixels becomes increasingly concentrated over successive iterations. In other words, as the attack progresses, the perturbation tends to focus on a progressively smaller subset of highly sensitive regions.

This phenomenon indicates that the importance distribution of input pixels is not static, but evolves dynamically during optimization. Early iterations may require a broader set of pixels to explore the loss landscape, while later stages naturally refine the perturbation toward a more compact and targeted subset. Such behavior suggests that maintaining a fixed sparsity level may be suboptimal, as it does not fully exploit the changing structure of gradient information.

Motivated by this insight, a potential extension of this work is to develop a mechanism that dynamically adjusts the number of selected pixels across iterations. This could lead to more efficient attacks by reducing unnecessary updates in later stages, while preserving strong attack performance. Moreover, adaptive sparsity may provide a better balance between attack effectiveness and perceptual quality, as it inherently aligns with the evolving importance of input dimensions.

Exploring principled strategies for dynamic pixel selection, as well as understanding their theoretical properties, remains an open and compelling direction for future work.

---

## 9. Conclusion

In conclusion, this work critically revisits the fundamental mechanics of adversarial attacks through the lens of spatial sparsity. Traditional first-order gradient methods operate under the implicit assumption that maximizing adversarial loss requires pervasive, dense perturbations distributed across the entire continuous input space. By systematically analyzing the role of pixel-level importance, we challenge this prevailing paradigm and demonstrate that adversarial vulnerability is, in fact, profoundly concentrated in highly localized semantic regions.

Our proposed gradient-guided selective update mechanism successfully leverages this non-uniform spatial vulnerability, proving that catastrophic adversarial deception can be achieved while leaving the vast majority of the original image pristine. Extensively evaluated on both standard and adversarially trained models, our approach not only achieves attack success rates that rival standard dense methods but also significantly enhances the perceptual fidelity, visual stealth, and interpretability of the generated adversarial examples.

Ultimately, this research bridges the critical gap between adversarial efficacy and spatial efficiency. By demonstrating that effective adversarial perturbations can be constructed without modifying the entire input manifold, our findings open up highly promising new directions for developing structured, interpretable, and physically realizable adversarial attacks. Concurrently, these insights provide a crucial foundation for designing more targeted, region-aware defense mechanisms against spatially localized threats in deep neural networks.

---

## References

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). *Explaining and harnessing adversarial examples*. arXiv preprint arXiv:1412.6572.
2. Kurakin, A., Goodfellow, I., & Bengio, S. (2016). *Adversarial examples in the physical world*. arXiv preprint arXiv:1607.02533.
3. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). *Towards deep learning models resistant to adversarial attacks*. arXiv preprint arXiv:1706.06083.
4. Papernot, N., McDaniel, P., Jha, S., Fredrikson, M., Celik, Z. B., & Swami, A. (2016). *The limitations of deep learning in adversarial settings*. In *2016 IEEE European symposium on security and privacy (EuroS&P)* (pp. 372--387). IEEE.
5. Su, J., Vargas, D. V., & Sakurai, K. (2019). *One pixel attack for fooling deep neural networks*. *IEEE Transactions on Evolutionary Computation*, 23(5), 828--841.
6. Modas, A., Moosavi-Dezfooli, S. M., & Frossard, P. (2019). *Sparsefool: a few pixels make a big difference*. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition* (pp. 9087-9096).
7. Dong, X., Chen, D., Bao, J., Qin, C., Yuan, L., Zhang, W., ... & Chen, D. (2020). *Greedyfool: Distortion-aware sparse adversarial attack*. Advances in neural information processing systems, 33, 11226-11236.
8. Zhong, Y., & Liu, S. (2021). *Sparse-PGD: L0-bounded PGD for sparse adversarial perturbations*.
9. Croce, F., & Hein, M. (2020). *Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks*. In *International conference on machine learning* (pp. 2206-2216). PMLR.