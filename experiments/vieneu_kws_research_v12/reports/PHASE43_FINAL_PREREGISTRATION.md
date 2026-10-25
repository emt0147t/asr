# PHASE 43: LOCKED INDEPENDENT EVALUATION PROTOCOL & PRE-REGISTRATION

**Repository:** `D:\trainidentificationaudio`  
**Phase:** Phase 43 (Locked Inter-Generator Evaluation Protocol)  
**Creation Timestamp:** `2026-10-25T13:12:25.782927+00:00`  
**Manifest Freeze Timestamp:** `2026-10-25T13:13:09.209714+00:00`  
**Status:** **LOCKED & PRE-REGISTERED (PRIOR TO EVALUATION INFERENCE)**

---

## 1. Governance & Protocol Invariants

1. **Default Model Status:** Legacy $v11$ remains the canonical default model.
2. **Model Integrity:** Checkpoints for $v11$, $E1$, and Phase 28 $D_2\_H1\_seed42$ are strictly read-only and immutable.
3. **Threshold Lock:** Decision threshold $\theta_{\text{cmd}} = 0.20$ is strictly locked.
4. **Argmax Policy:** REJECT-inclusive 5-class argmax:
   $$\hat{c} = \begin{cases} \arg\max_{i \in \{0,1,2,3\}} P(c_i), & \text{if } \max_{i \in \{0,1,2,3\}} P(c_i) \ge 0.20 \\ 4 \text{ (REJECT)}, & \text{otherwise} \end{cases}$$
5. **VAD Specification:** `SpeechBandRatioVAD` with band $[300, 3400]\text{ Hz}$, energy threshold $-46.0\text{ dBFS}$, ratio threshold $0.32$.
6. **Frontend:** 40-channel Log-Mel filterbank ($16\text{ kHz}$, frame $25\text{ ms}$, hop $10\text{ ms}$, $512$ FFT).
7. **Capacity:** Fixed at $138,469$ parameters (`CorrectedVietnameseKWSNet`, `DEFER_E2_CAPACITY`).
8. **Deployment Claim Restrictions:**
   - `EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED`
   - `NOT_VALIDATED_ON_NATURAL_SPEECH`
   - `NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION`
   - `NOT_VALIDATED_PENDING_DATA`

---

## 2. Independent Corpus Provenance & Hashes

### 2.1 Acoustic Generators & Noise Assets
- **Acoustic Speech Generator:** `facebook/mms-tts-vie` (Meta AI VITS architecture, HuggingFace commit `b58928d0...`).
- **Environmental Noise:** Karol Piczak ESC-50 Fold 1 indoor categories (`washing_machine`, `vacuum_cleaner`, `keyboard_typing`, `clock_tick`). Fully verified non-overlapping with historical noise ($r_{\max} < 0.12$).
- **Multi-Instance Strategy:** 4 distinct synthesis instances per lexeme (varying seed and speaking rate $\in [0.85, 1.15]$), solving the $N_{\text{text}}=1$ cluster limitation identified in Phase 40.

### 2.2 Manifest SHA-256 Ledger

| Manifest Artifact | File Path | Total Items | SHA-256 Checksum |
|:------------------|:----------|:-----------:|:-----------------|
| **Isolated Test Manifest** | `experiments/vieneu_kws_research_v12/manifests/PHASE43_LOCKED_MANIFEST.csv` | 480 clips | `179d173f2d4347b8d6030c7bb0a5d61179e8100b306577958e0d93288e92e4ee` |
| **Tone Probe Manifest** | `experiments/vieneu_kws_research_v12/manifests/PHASE43_TONE_PROBE_MANIFEST.csv` | 96 clips | `913644e38644d7a8f72fe953e6815e4004af85b707396c0fa2de11f708199880` |
| **Streaming Test Manifest** | `experiments/vieneu_kws_research_v12/manifests/PHASE43_STREAMING_MANIFEST.csv` | 50 sequences | `4681bdbe11da6c25597787588790f7bf4ac7e1f7957b4c3a1ba30b6b3e90981b` |
| **Streaming Event Ledger (CSV)** | `experiments/vieneu_kws_research_v12/manifests/PHASE43_EVENT_LEDGER.csv` | 211 events | `84c9af4247b091279149b3ef0dda8ebdbafabbd7679f4f6f465d7064c582c3cb` |
| **Streaming Event Ledger (JSON)** | `experiments/vieneu_kws_research_v12/manifests/PHASE43_EVENT_LEDGER.json` | 50 sequences | `7736b3b13e85d1db9772c2b084dae9019f516c8fcfa56f329f12cb860982ef0c` |

---

## 3. Pre-Registered Metrics & Statistical Protocol

### 3.1 Primary Isolated Metrics
1. **4-Command Macro-F1:** Unweighted mean of $F_1$ across `TRAI`, `PHAI`, `LEN`, `XUONG`.
2. **5-Class Macro-F1:** Unweighted mean of $F_1$ across all 5 classes (including `REJECT`).
3. **Phonetic OOV False Alarm Rate:** FAR on phonetic hard negative cohort ($N = 256$).
4. **General OOV False Alarm Rate:** FAR on general everyday OOV cohort ($N = 160$).
5. **Stratified Slices:** SNR tiers (`clean`, `15 dB`, `10 dB`, `5 dB`), Speaking Rate (`slow`, `normal`, `cadenced`, `fast`), and Synthesis Instance.

### 3.2 Primary Streaming Metrics
1. **Exact Sequence Accuracy:** Fraction of 50 sequences where predicted command sequence matches ground truth exactly.
2. **Event Metrics:** Precision, Recall, $F_1$ with temporal matching window $\pm 0.50\text{ s}$ around event midpoint.
3. **Continuous Exposure Denominators:**
   - Pure Silence/Noise FA/min (10 sequences, $\sim 100\text{ s}$ total exposure).
   - Pure OOV Conversation FA/min (10 sequences, $\sim 100\text{ s}$ total exposure).
4. **Zero-Numerator Upper 95% Confidence Bounds (Rule of Three):**
   - When 0 false alarms are observed over $N$ opportunities:
     $$\text{UCB95} = 1 - (1 - 0.95)^{1/N} \approx \frac{3}{N}$$
   - When 0 false alarms are observed over $T_{\text{min}}$ minutes of continuous audio:
     $$\text{UCB95} = \frac{3}{T_{\text{min}}} \text{ FA/min}$$

### 3.3 Bootstrap Resampling Hierarchy
- Estimand A: Observed point estimate difference.
- Estimand B: Lexeme Cluster Bootstrap (30 base text clusters).
- Estimand C: Synthesis Instance Cluster Bootstrap (120 groups: 30 texts $\times$ 4 synthesis instances).
- Estimand D: Clip-level i.i.d. bootstrap ($N = 480$).
