# dkf

## Author Contributions

| Name | Main Contributions | Credit |
|------|--------------------|--------|
| **Lingpeng Chen** | • Architected the neural-network components<br>• Implemented the full training & inference pipeline | 50% |
| **Hongxu Zhao**   | • Led the mathematical derivations and theoretical analysis | 50% |

Both authors jointly conceived **Kalman-Guided Deep Kalman Filters (KG-DKF)** and continue to co-develop the project.

---

## Project Overview

KG-DKF couples classical analytic filtering with modern deep variational smoothing to deliver **robust, data-efficient robot state estimation**.

* Combines a learnable Deep Kalman Filter with an explicit Kalman‐style guidance term.  
* Retains the closed-form uncertainty propagation of the Kalman Filter while leveraging deep networks to model complex, non-linear dynamics.  
* Achieves higher accuracy with significantly fewer supervised samples compared to purely data-driven baselines.

We are actively:

1. Refining the algorithmic design and training strategies.  
2. Adding extensive benchmarks on simulated and real robotic platforms.  
3. Quantifying robustness under heavy noise, missing data, and abrupt dynamics changes.  
4. Profiling runtime to demonstrate the efficiency gains of our hybrid framework.

Stay tuned for upcoming releases and results!