# Methodological Plan: Optional 5-Clip Inter-Rater Reliability Subset

**Workspace:** `D:\trainidentificationaudio`  
**Experiment Root:** `experiments/vieneu_kws_research_v10/human_review/distributed_batches/`  
**Protocol Status:** `DOCUMENTED_PLAN_ONLY` (Default packages maintain strictly 20 unique clips)  
**Governance Banner:** `PENDING_HUMAN_LISTENER_VERIFICATION` | `NOT_VALIDATED_ON_NATURAL_SPEECH`  
**Date:** 2026-10-18  

---

## 1. Methodological Limitation of the 20-Unique-Clip Allocation

In the current distributed listening review design:
- 120 total audio clips are partitioned into 6 disjoint sets of 20 clips:
  $$\mathcal{S} = \bigcup_{k=1}^6 \mathcal{S}_k, \quad \text{where } |\mathcal{S}_k| = 20 \quad \text{and} \quad \mathcal{S}_i \cap \mathcal{S}_j = \emptyset \text{ for } i \ne j$$
- Each audio clip is evaluated by **exactly one reviewer** ($R = 1$ rating per clip).

### Scientific Consequences:
1. **Zero Inter-Rater Reliability Estimation:**
   - Because no two reviewers listen to the same clip, it is mathematically impossible to compute inter-rater agreement statistics:
     - **Cohen’s Kappa ($\kappa$)** or **Fleiss’ Kappa ($\kappa$)** for categorical pronunciation decisions (`Y` / `N` / `Unsure`).
     - **Krippendorff’s Alpha ($\alpha$)** for nominal or ordinal ratings.
     - **Intraclass Correlation Coefficient ($ICC(2, 1)$ or $ICC(3, 1)$)** for continuous/ordinal Likert ratings (Intelligibility 1–5, Naturalness 1–5).
2. **Confounding of Rater Severity and Audio Quality:**
   - If Reviewer 1 reports a mean naturalness of $3.1$ and Reviewer 2 reports $4.4$, one cannot determine whether Reviewer 1 received acoustically inferior clips or whether Reviewer 1 applies a stricter grading standard.
   - Reviewer bias cannot be adjusted or normalized across unique disjoint batches.

---

## 2. Proposed 5-Clip Common Reliability Protocol

To enable rigorous statistical inter-rater agreement without substantially increasing human workload, we propose an optional **5-clip common control subset** ($N_{\text{common}} = 5$).

### A. Recommended Selection Criteria for Common Anchor Clips
The 5 common clips should be sampled across the difficulty and phonetic spectrum:
1. **Anchor 1 (Clear Command):** A high-confidence synthesized command (e.g. `LEN` or `TRAI` by `Thùy Dung`). Tests high-ceiling agreement.
2. **Anchor 2 (Minimal Pair Hard-Negative):** A tone-confusable negative (e.g. `xưởng in` or `cai trị`). Tests false-acceptance boundary agreement.
3. **Anchor 3 (Phrase Context Hard-Negative):** A compound phrase (e.g. `lên sàn` or `phạt thẻ`). Tests multi-syllable boundary agreement.
4. **Anchor 4 (Acoustic Silence/Ambient):** A real room ambient clip (e.g. `office_ambient`). Tests baseline quiet detection.
5. **Anchor 5 (Acoustically Marginal Synthetic Clip):** A clip with mild tonal flattening or synthesized buzzing. Tests rater sensitivity to synthetic speech artifacts.

### B. Package Structure under Reliability Mode
If activated:
- Each reviewer package contains **25 clips** ($20\text{ unique} + 5\text{ common}$).
- The 5 common clips are given neutral randomized local identifiers (`clip_01` through `clip_25`) interleaved within each package so reviewers cannot identify which clips are shared controls.
- Total unique clips evaluated across all reviewers remains 120. Total evaluations across the campaign increases from $120$ to $150$ ($6 \times 25$).

---

## 3. Statistical Analysis Enabled by the 5-Clip Subset

With 6 reviewers rating the same 5 common clips:

1. **Ordinal / Likert Intelligibility & Naturalness:**
   - Calculate two-way random effects Intraclass Correlation Coefficient $ICC(2, k)$ and $ICC(2, 1)$:
     $$ICC(2, 1) = \frac{MS_R - MS_E}{MS_R + (k - 1)MS_E + \frac{k}{n}(MS_C - MS_E)}$$
     where $MS_R$ is mean square between clips, $MS_C$ is mean square between reviewers, and $MS_E$ is error variance.
2. **Categorical Pronunciation (`Y` / `N` / `Unsure`):**
   - Calculate Fleiss' Kappa ($\kappa$) across the 6 raters on the common items to confirm whether native listeners agree on phoneme/tone correctness.
3. **Rater Severity Calibration:**
   - Standardize each reviewer's ratings using their mean and standard deviation on the 5 common clips:
     $$z_{i, k} = \frac{x_{i, k} - \mu_{k, \text{common}}}{\sigma_{k, \text{common}}}$$
     allowing unconfounded pooling of the 20 unique clips.

---

## 4. Current Campaign Decision

**The 5-clip common subset is NOT activated by default.**  
The primary campaign strictly packages **20 unique non-overlapping clips per reviewer** as specified by project requirements. If the project team approves expanding reviewer workload from 20 to 25 clips, this documented plan can be triggered by running `create_distributed_batches.py --include_common_subset`.
