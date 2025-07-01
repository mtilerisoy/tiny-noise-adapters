# Project Card -  **Lightweight Noise-Robust Adaptation of Audio Foundation Models**

---

**Problem Statement**

Self-supervised encoders for heart- and lung-sound analysis are pre-trained on carefully collected, low-noise, clinical datasets. In a real-world scenario, there will be unexpected noises, microphone artefacts; therefore, performance is expected drop sharply.

**Proposed Idea**

Keep the SSL-encoder **frozen** and learn a **lightweight, architecture-agnostic adapter** that maps noisy embeddings back to the clean manifold. Training relies solely on synthetically noised copies of the original dataset – no re-training of the encoder or task-specific head required.

**Hypothesis**

A feature-transformation space can be learned for an encoder such that the noise within an input sample can be cleaned through linear projections of intermediate features.

In other words there exists a  *linear projection* * that can transform a noisy embedding space into its clean version.

**Assumptions**

1. Everyday sounds and AugLy augmentation span the real-world noise
2. Linear projection of intermediate features is enough for cleaning the noise
3. Limited labelled data will be enough for comprehensive evaluation
   1. can be collected from volunt. for simple tasks (low-priority)

**Research questions**

1. Which type of noise and SNR levels (and/or combinations) are harmful for the foundation models?
2. How do other fine-tuning or adaptation methods compare in terms of performance, training duration, inference efficiency, and parameter counts?
   1. Frequency filters
   2. ~PEFT~ → FM will loose generalizability (needs labelled data for a task) && This will basically be our method
   3. denoising encoder/decoder
   4. other noise suppression techniques
3. How much of the clean data performance is traded for robustness?

---

## Design Choices

**Noise Types**

* **Ambient:** Background noise coming from external sources that can interfere with the target audio
  * Background chatter, keyboard clicks, traffic sound etc.
* **Sonic Artefact:** Noise appearing within the target audio due to hardware or user-related problems
  * Blocked or deformed microphone, mic being too far, bit-rate drops, jitter, loudness etc.

**Noise Levels (SNR Ladder)**

SNR levels of {5, 10, 15, 20, +inf} where +inf is the original clean sound.

* too bad recordings (-5 or 0 SNR) might be ablations (stay realistic)
* Batch Creation
  * Have 2-3 sampling strategies based on noise levels
  * we can even have clean samples
  * is curriculum lr working
  * is smart batch creation working

---

### Action Plan & Timeline

| Phase                                                    | Success criteria                                                                        |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Noise protocol creation<br />Synthetic dataset** | Fully reproducible noise-injection script<br />Generate the testing dataset             |
| **Baseline evaluation** (OPERA-CE)                 | Re-produce original performance<br />Noisy dataset performance degradation plots/tables |
| **Adapter Design & Implementation**                | Implement proposed solution                                                             |
| **Training**                                       |                                                                                         |
| **Ablations**                                      | Paper-ready charts & tables                                                             |

---

### Ablations

| What                          | How                                                                                                                    | Question answered                                                     |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **Adapter type**        | LoRA vs Prompt vs Ours                                                                                                 | Which parameter-efficient scheme is the best and how do they compare? |
| **Adaptation location** | Only after encoder vs multiple mid-layers                                                                              | Does modifying intermediate features help?                            |
| **Training objective**  | (a) Sup. cross-entropy; (b) Contrastive distillation; (c) CKA loss for distribution alignment; (d) Curriculum learning | Does matching clean/noisy distributions outperform CE?                |

How does curriculum lr affect training and convergence |
| **Noise diversity** | Trained on (i) ambient only, (ii) artefact only, (iii) both | Does mixed-noise training generalise better? |
| **Cross-domain** | Train on cardiac, test on respiratory (and vice-versa) | Is a **single-modal** adapter transferrable? |

---

### Contributions

1. **Synthetic Noise-Injection Module**
2. **Synthetic Noisy Dataset** (bigger than the clean set)
3. **Baseline Leaderboard** of audio FMs on noisy benchmark
4. **Lightweight Noise Adapter** (plug-and-play modules)
5. **Comprehensive Benchmark Report** (ablations)

---

</aside>
