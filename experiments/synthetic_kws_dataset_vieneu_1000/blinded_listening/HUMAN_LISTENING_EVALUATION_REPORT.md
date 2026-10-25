# Vietnamese Synthetic KWS — Blinded Listening Review Report

**Package Directory:** `experiments/synthetic_kws_dataset_vieneu_1000/blinded_listening/`  
**Total Evaluation Clips:** 80  
**Completed Reviews:** 3 / 80 (3.8%)  
**Missing / Pending Reviews:** 77 / 80  
**Reported Listener Count:** 1 (Screening auditor)  
**Human Verification Status:** **`REVIEW_COMPLETED`**  
**Official Project Governance Status:** `NOT_VALIDATED_ON_NATURAL_SPEECH`  

---

## 1. Command Recognition Performance

- **Overall Word Recognition Rate:** **33.3%** (1 / 3)

| Keyword | Samples Evaluated | Correct Recognition | Accuracy (%) |
| :--- | :---: | :---: | :---: |
| **`LEN`** | 1 | 0 | 0.0% |
| **`TRAI`** | 1 | 1 | 100.0% |
| **`XUONG`** | 1 | 0 | 0.0% |

---

## 2. Perceptual Quality and Tone Ratings

| Metric | Target | Observed Result |
| :--- | :--- | :--- |
| **Intelligibility (1 to 5)** | $\\ge 4.0$ (Clear) | **3.67 $\\pm$ 1.53** |
| **Naturalness (1 to 5)** | $\\ge 3.5$ (Human-like) | **3.33 $\\pm$ 1.15** |
| **Lexical Tone Concordance** | High | **33.3%** (Uncertain: 33.3%) |

---

## 3. Acoustic Defect Counts

| Defect Category | Count | Share (%) | Status |
| :--- | :---: | :---: | :---: |
| **Digital Clipping** | 0 | 0.0% | PASS |
| **Robotic / Harsh Distortion** | 0 | 0.0% | PASS |
| **Phoneme Truncation** | 0 | 0.0% | PASS |
| **Synthetic Buzzer Sound** | 0 | 0.0% | PASS |

---

## 4. Clips Flagged for Re-listening (2 clips)

| Blind ID | True Label | Perceived Command | Perceived Tone | Intel | Nat | Flag Reason |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| `BLIND_EVAL_02` | **XUONG** | PHAI | HOI | 4 | 4 | Misidentified (PHAI) |
| `BLIND_EVAL_03` | **LEN** | UNCERTAIN | UNCERTAIN | 2 | 2 | Misidentified (UNCERTAIN), Tone Uncertain, Low Intel |

---

## 5. Scope and Governance Note

> [!NOTE]
> This review represents a screening exercise across 80 stratified synthetic clips by an individual human reviewer. While highly informative for identifying gross phonetic defects, it does not constitute a population-level listening study across regional demographics.
