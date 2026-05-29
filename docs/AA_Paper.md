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

* **Jacobian-based Saliency Map Approach (JSMA):** Shifting focus towards extreme sparsity, Papernot et al. introduced JSMA. This method constructs a saliency map by computing the forward derivative (Jacobian matrix) of the network, explicitly identifying which specific pixels exert the highest influence on the target class probabilities. The algorithm iteratively perturbs these highly salient pixels. While highly effective at minimizing the $L_0$ norm, the repeated computation of the full Jacobian matrix renders JSMA computationally prohibitive and difficult to scale to high-resolution image data.
* **One-pixel Attack:** Pushing the boundaries of spatial sparsity, Su et al. demonstrated that deep neural networks can be catastrophically deceived by altering just a single pixel. Instead of relying on gradient information, this method employs Differential Evolution—a heuristic population-based search algorithm—to find the optimal pixel coordinates and color values. Although successful in black-box settings, the evolutionary search process requires extensive model queries, rendering it highly inefficient and fundamentally different from gradient-guided optimization strategies.

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

To strictly enforce the $L_0$ constraint, we introduce a non-differentiable binary projection function, $\Phi(\cdot)$, which maps the continuous importance scores onto a sparse Boolean mask $M_t \in \{0, 1\}^n$:

$$M_t = \Phi(S_t, k)$$

where $\Phi$ acts as a spatial filter (e.g., a $\text{Top-}k$ operator yielding $1$ for the $k$ indices with the largest values in $S_t$, and $0$ elsewhere). This modular formulation decouples the spatial selection mechanism from the perturbation generation. It effortlessly accommodates different strategic selection criteria, including static masks (computed strictly at $t=0$) or dynamic tracking masks (recalculated at each iteration), leaving the refinement of $\Phi$ as an open dimension for future architectures.

### 4.3 Iterative Masked Projection Update

With the sparse feasible subspace explicitly defined by $M_t$, we restrict the adversarial updates strictly to the selected regions. We modify the standard iterative update rule by incorporating the binary mask via the Hadamard product ($\odot$):

$$x_{t+1} = \Pi_{\mathcal{B}_\epsilon(x) \cap \mathcal{V}} \left( x_t + \alpha \left( M_t \odot \text{sign}(g_t) \right) \right)$$

Here, $\alpha$ denotes the optimization step size, $\Pi$ is the projection operator, $\mathcal{B}_\epsilon(x)$ represents the $L_\infty$ $\epsilon$-ball centered around the original clean input $x$, and $\mathcal{V} = [0, 1]^n$ bounds the valid continuous image space. By masking the gradient step, this update completely freezes the unselected dimensions ($1 - M_t$), thereby inherently satisfying the $L_0 \ll n$ sparsity constraint at every optimization iteration while locally maximizing the adversarial loss.

### 4.4 Interpretation: Attacks as Spatial Feature Selection

This formulation introduces a fundamental paradigm shift in the conceptualization of adversarial generation. Traditional first-order adversaries, such as standard PGD, optimize over the entire unconstrained image manifold, treating the input space as a uniform canvas for noise addition. In contrast, our approach reframes adversarial attacks as a constrained submodular optimization problem over the input features. By confining the perturbation $\delta$ to a lower-dimensional subspace spanned by the most salient pixels, we empirically demonstrate that successful adversarial deception does not require pervasive noise. Instead, network fragility is heavily concentrated in highly localized semantic regions, allowing us to disentangle true adversarial vulnerability from dense pixel manipulation.

---

## 5. Experiments

### 5.1 Experimental Setup

We evaluate the proposed method on the CIFAR-10 dataset, which consists of 60,000 images across 10 classes. Two types of models are considered in our experiments:

1. **Standard model:** Trained on CIFAR-10 using conventional empirical risk minimization.
2. **Robust model:** Obtained via adversarial training on CIFAR-10, designed to improve resistance against gradient-based attacks.

All experiments are conducted on a subset of 1,000 test images to ensure consistency and fair comparison across different attack methods. We compare our approach with several widely used gradient-based attacks, including FGSM, BIM, and PGD.

To comprehensively evaluate attack performance, we report multiple metrics, including Attack Success Rate (ASR), classification accuracy, sparsity (i.e., the proportion of modified pixels), Structural Similarity Index (SSIM), and Peak Signal-to-Noise Ratio (PSNR).

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



### 5.3 Results on Standard Model

**Table 1:** Comparison of attack performance on the standard CIFAR-10 model. Sparse attacks achieve competitive attack success rates while modifying significantly fewer pixels and preserving higher perceptual quality.

| Attack | K | Iter | Acc (%) | ASR (%) | $L_0$ | Sparsity (%) | $L_2$ | $L_\infty$ | SSIM | PSNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clean | - | 0 | 94.8 | 0.0 | 0 | 100.0 | 0.000 | 0.000 | 1.0000 | $\infty$ |
| FGSM | - | 1 | 30.5 | 68.0 | 1021 | 0.3 | 1.724 | 0.031 | 0.9380 | 30.14 |
| BIM | - | 10 | 0.0 | 100.0 | 997 | 2.6 | 1.209 | 0.031 | 0.9692 | 33.22 |
| PGD | - | 10 | 0.0 | 100.0 | 1023 | 0.1 | 1.246 | 0.031 | 0.9666 | 32.96 |
| **Sparse** | 0.1 | 10 | 8.2 | 91.5 | 400 | 60.9 | 0.418 | 0.031 | **0.9947** | **42.41** |
| **Sparse** | 0.2 | 10 | 1.9 | 98.0 | 642 | 37.3 | 0.575 | 0.031 | 0.9906 | 39.66 |
| **Sparse** | 0.3 | 10 | 0.6 | 99.4 | 786 | 23.2 | 0.686 | 0.031 | 0.9872 | 38.12 |
| **Sparse** | 0.4 | 10 | 0.3 | 99.7 | 876 | 14.4 | 0.774 | 0.031 | 0.9843 | 37.08 |
| **Sparse** | 0.5 | 10 | 0.2 | 99.8 | 933 | 8.9 | 0.849 | 0.031 | 0.9818 | 36.28 |
| **Sparse** | 0.6 | 10 | 0.1 | 99.9 | 969 | 5.4 | 0.913 | 0.031 | 0.9797 | 35.65 |
| **Sparse** | 0.7 | 10 | 0.0 | 100.0 | 991 | 3.2 | 0.968 | 0.031 | 0.9778 | 35.14 |
| **Sparse** | 0.8 | 10 | 0.0 | 100.0 | 1001 | 2.2 | 1.061 | 0.031 | 0.9740 | 34.36 |
| **Sparse** | 0.9 | 10 | 0.0 | 100.0 | 1012 | 1.2 | 1.154 | 0.031 | 0.9702 | 33.63 |
| **Sparse** | 1.0 | 10 | 0.0 | 100.0 | 1020 | 0.4 | 1.247 | 0.031 | 0.9664 | 32.96 |


Dense attacks such as PGD and BIM achieve near-perfect attack success rates (close to 100%), but at the cost of modifying almost all pixels in the input image. In contrast, the proposed method achieves comparable attack performance while modifying only a subset of pixels. Notably, even with a significantly reduced number of perturbed pixels, the attack remains highly effective.

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
| **Sparse** | 0.1 | 10 | 73.1 | 14.9 | 242 | 76.3 | 0.485 | 0.031 | 0.9973 | 41.15 |
| **Sparse** | 0.2 | 10 | 68.2 | 20.6 | 431 | 57.9 | 0.687 | 0.031 | 0.9946 | 38.14 |
| **Sparse** | 0.3 | 10 | 65.4 | 23.9 | 587 | 42.7 | 0.840 | 0.031 | 0.9919 | 36.39 |
| **Sparse** | 0.4 | 10 | 64.0 | 25.5 | 716 | 30.1 | 0.969 | 0.031 | 0.9895 | 35.15 |
| **Sparse** | 0.5 | 10 | 62.6 | 27.2 | 822 | 19.7 | 1.082 | 0.031 | 0.9871 | 34.19 |
| **Sparse** | 0.6 | 10 | 61.6 | 28.4 | 904 | 11.7 | 1.182 | 0.031 | 0.9848 | 33.42 |
| **Sparse** | 0.7 | 10 | 60.9 | 29.2 | 962 | 6.0 | 1.273 | 0.031 | 0.9825 | 32.78 |
| **Sparse** | 0.8 | 10 | 60.4 | 29.8 | 983 | 4.0 | 1.399 | 0.031 | 0.9771 | 31.96 |
| **Sparse** | 0.9 | 10 | 59.8 | 30.4 | 1002 | 2.1 | 1.526 | 0.031 | 0.9718 | 31.20 |
| **Sparse** | 1.0 | 10 | 59.3 | 31.0 | 1017 | 0.7 | 1.653 | 0.031 | 0.9665 | 30.51 |


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
* We provide robust empirical proof through rigorous, extensive experiments, confirming that these localized, sparse perturbations maintain highly competitive attack success rates compared to traditional dense methods.
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