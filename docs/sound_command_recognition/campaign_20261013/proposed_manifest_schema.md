# Proposed Manifest Schema for Human Field Speech Data

> **Document Version:** 1.0.0-PROPOSAL  
> **Campaign Date:** October 13, 2026  
> **Target Manifest:** `data/raw/human_recordings_v1/human_collection_manifest_v1.csv`  
> **Companion Ledger:** `data/raw/human_recordings_v1/consent_ledger_human.json`

---

## 1. Schema Overview & Design Principles

To ensure complete traceability, auditability, and protection against data leakage, the human field collection campaign adopts an expanded 28-column relational schema.

### Core Design Principles:
1. **Cryptographic Traceability:** Tracks both raw original and standardized audio checksums (`sha256_raw`, `sha256_standardized`).
2. **Explicit Provenance:** Strictly disallows vague provenance strings. Every entry must declare an explicit provenance enum (`HUMAN_CONSENTED`, `SYNTHETIC_TEST_FIXTURE`, `UNVERIFIED_OR_TEST_FIXTURE`).
3. **Multi-Class Support:** Natively catalogs `TRAI`, `PHAI`, `LEN`, `XUONG`, `UNKNOWN`, and `SILENCE` with acoustic sub-categories.
4. **Disjoint Partitioning Enablers:** Explicitly tracks `speaker_id`, `session_id`, `device_model`, and `distance_category` to ensure zero leakage across splits.
5. **PII Elimination:** Contains zero personal identifying information. Links exclusively to pseudonymous contributor IDs verified in the companion consent ledger.

---

## 2. Complete 28-Column Data Dictionary

| # | Column Name | Type | Nullable | Permitted Values / Regex | Description & Verification Rule |
| :-: | :--- | :---: | :---: | :--- | :--- |
| **1** | `recording_id` | String | No | `^rec_h_[a-z0-9_\-]+_take\d{3}$` | Primary Key. Structured globally unique identifier. |
| **2** | `source` | String | No | `vietnamese_kws_human_campaign_v1` | Identifies the physical field recording campaign. |
| **3** | `source_version` | String | No | `v1.0.0-human` | Release iteration tag. |
| **4** | `license` | String | No | `Proprietary-Consented-Research-Commercial` | Approved SPDX / legal agreement identifier. |
| **5** | `consent_id` | String | No | `^CONSENT_H_[NCS]_[a-f0-9]{6}_v1\.0$` | Cryptographic reference to signed consent entry. |
| **6** | `speaker_id` | String | No | `^SPK_H_[NCS]_[a-f0-9]{6}$` | Foreign Key to `consent_ledger_human.json`. |
| **7** | `dialect` | Enum | No | `Northern` \| `Central` \| `Southern` | Regional dialect group verified during registration. |
| **8** | `gender` | Enum | No | `FEMALE` \| `MALE` \| `OTHER` | Self-reported acoustic gender category. |
| **9** | `age_bracket` | Enum | No | `18-30` \| `31-50` \| `51+` | Broad age demographic bracket. |
| **10** | `command_label` | Enum | No | `TRAI` \| `PHAI` \| `LEN` \| `XUONG` \| `UNKNOWN` \| `SILENCE` | Categorical 6-class target label. |
| **11** | `class_id` | Integer| No | `0` \| `1` \| `2` \| `3` \| `4` \| `5` | 0=TRAI, 1=PHAI, 2=LEN, 3=XUONG, 4=UNKNOWN, 5=SILENCE. |
| **12** | `transcript` | String | No | UTF-8 Vietnamese text | Exact normalized transcript of the spoken utterance. |
| **13** | `sub_category` | Enum | No | `isolated_command`<br/>`confusing_near_rhyme`<br/>`carrier_sentence_embedded`<br/>`conversational_speech`<br/>`background_room_tone`<br/>`mechanical_fan_hvac`<br/>`device_handling_noise` | Fine-grained acoustic category distinguishing target isolated words, confusing negative phonemes, embedded sentences, and silence types. |
| **14** | `session_id` | String | No | `^SES_H_[a-zA-Z0-9_\-]+_\d{2}$` | Specific capture session block. |
| **15** | `device_model` | String | No | e.g., `iPhone 13`, `Galaxy S22`, `Redmi Note 11` | Hardware capture device model. |
| **16** | `microphone_type` | Enum | No | `builtin_mobile`<br/>`wired_headset`<br/>`bluetooth_earbuds`<br/>`usb_desktop_mic` | Microphone transducer configuration. |
| **17** | `environment` | Enum | No | `quiet_room`<br/>`living_room_reverberant`<br/>`office_hvac`<br/>`street_noise`<br/>`fan_noise` | Acoustic capture space. |
| **18** | `distance_category` | Enum | No | `close_talk_15cm`<br/>`desk_distance_60cm`<br/>`far_field_2m` | Distance from participant's mouth to microphone. |
| **19** | `speaking_style` | Enum | No | `normal` \| `fast` \| `slow` \| `quiet` \| `loud` \| `ambient_silence` | Dynamic delivery style. |
| **20** | `raw_duration_sec` | Float | No | $0.25 \le t \le 3.50$ | Exact duration of raw original recording in seconds. |
| **21** | `rms_db` | Float | No | $\text{Float} \le 0.0$ dBFS | Root-Mean-Square acoustic energy level in dBFS. |
| **22** | `label_provenance`| Enum | No | `HUMAN_CONSENTED`<br/>`SYNTHETIC_TEST_FIXTURE`<br/>`UNVERIFIED_OR_TEST_FIXTURE` | Primary forensic provenance class. `HUMAN_CONSENTED` requires verified consent entry. |
| **23** | `raw_filepath` | String | No | Relative path (e.g., `data/raw/...`) | Untouched raw file preserved with original container (`.wav`, `.m4a`). |
| **24** | `processed_path` | String | No | Relative path (e.g., `data/processed/...`) | Standardized 16 kHz mono 16-bit PCM 1.0s WAV copy. |
| **25** | `sha256_raw` | String | No | 64-char uppercase hex | SHA-256 checksum of `raw_filepath`. |
| **26** | `sha256_standardized`| String| No | 64-char uppercase hex | SHA-256 checksum of `processed_path`. |
| **27** | `status` | Enum | No | `APPROVED` \| `QUARANTINED` \| `REVOKED_WITHDRAWN` | Quality gate status. |
| **28** | `exclusion_reason`| String | Yes | Empty if APPROVED; failure reason if QUARANTINED | Specific rejection code (e.g., `digital_clipping`, `abnormal_silence`). |

---

## 3. Schema Invariants & Validation Rules

Any automated tool ingesting or auditing this manifest must enforce the following invariants:
1. **Primary Key Uniqueness:** $\text{Count}(\text{Distinct } \texttt{recording\_id}) = \text{Total Rows}$.
2. **Foreign Key Integrity:** For every row with $\texttt{label\_provenance} = \texttt{HUMAN\_CONSENTED}$, the $\texttt{speaker\_id}$ MUST exist in `consent_ledger_human.json` with $\texttt{consent\_given} = \texttt{true}$.
3. **File System Parity:**
   - $\texttt{raw\_filepath}$ must exist on disk and its SHA-256 must equal $\texttt{sha256\_raw}$.
   - $\texttt{processed\_path}$ must exist on disk and its SHA-256 must equal $\texttt{sha256\_standardized}$.
4. **Standardization Format Adherence:**
   - Every file referenced by $\texttt{processed\_path}$ must have:
     $$\text{Sample Rate} = 16,000 \text{ Hz}, \quad \text{Channels} = 1, \quad \text{Subtype} = \texttt{PCM\_16}, \quad \text{Frames} = 16,000 \text{ (1.000 s)}$$
5. **Class ID vs. Label Mapping Integrity:**
   - $\texttt{class\_id} = 0 \iff \texttt{command\_label} = \texttt{TRAI}$
   - $\texttt{class\_id} = 1 \iff \texttt{command\_label} = \texttt{PHAI}$
   - $\texttt{class\_id} = 2 \iff \texttt{command\_label} = \texttt{LEN}$
   - $\texttt{class\_id} = 3 \iff \texttt{command\_label} = \texttt{XUONG}$
   - $\texttt{class\_id} = 4 \iff \texttt{command\_label} = \texttt{UNKNOWN}$
   - $\texttt{class\_id} = 5 \iff \texttt{command\_label} = \texttt{SILENCE}$

---

## 4. Concrete Example Manifest Rows

Below are representative records demonstrating the schema across different classes:

```csv
recording_id,source,source_version,license,consent_id,speaker_id,dialect,gender,age_bracket,command_label,class_id,transcript,sub_category,session_id,device_model,microphone_type,environment,distance_category,speaking_style,raw_duration_sec,rms_db,label_provenance,raw_filepath,processed_path,sha256_raw,sha256_standardized,status,exclusion_reason
rec_h_trai_SPK_H_N_8f2a1b_SES_H_01_take001,vietnamese_kws_human_campaign_v1,v1.0.0-human,Proprietary-Consented-Research-Commercial,CONSENT_H_N_8f2a1b_v1.0,SPK_H_N_8f2a1b,Northern,FEMALE,18-30,TRAI,0,trái,isolated_command,SES_H_8f2a1b_01,iPhone 13,builtin_mobile,quiet_room,close_talk_15cm,normal,0.68,-18.42,HUMAN_CONSENTED,data/raw/human_recordings_v1/audio_raw/SPK_H_N_8f2a1b/SES_H_8f2a1b_01/rec_h_trai_SPK_H_N_8f2a1b_SES_H_01_take001.m4a,data/raw/human_recordings_v1/audio_16k_standardized/TRAI/rec_h_trai_SPK_H_N_8f2a1b_SES_H_01_take001.wav,8F2A1B01C2DFB8F1FD402E689E42D95DBDAA59F8CA64020559D219AE385ED2D6,62FFEF78A520156A3A2E7C30EF7CEE41FDF16A23F2AC24301BD1B393517F4985,APPROVED,
rec_h_unknown_SPK_H_N_8f2a1b_SES_H_01_take002,vietnamese_kws_human_campaign_v1,v1.0.0-human,Proprietary-Consented-Research-Commercial,CONSENT_H_N_8f2a1b_v1.0,SPK_H_N_8f2a1b,Northern,FEMALE,18-30,UNKNOWN,4,trại,confusing_near_rhyme,SES_H_8f2a1b_01,iPhone 13,builtin_mobile,quiet_room,close_talk_15cm,normal,0.61,-19.15,HUMAN_CONSENTED,data/raw/human_recordings_v1/audio_raw/SPK_H_N_8f2a1b/SES_H_8f2a1b_01/rec_h_unknown_SPK_H_N_8f2a1b_SES_H_01_take002.m4a,data/raw/human_recordings_v1/audio_16k_standardized/UNKNOWN/rec_h_unknown_SPK_H_N_8f2a1b_SES_H_01_take002.wav,1B01C2DFB8F1FD402E689E42D95DBDAA59F8CA64020559D219AE385ED2D68F2A,7CEE41FDF16A23F2AC24301BD1B393517F498562FFEF78A520156A3A2E7C30EF,APPROVED,
rec_h_unknown_SPK_H_S_a1d3f5_SES_H_01_take015,vietnamese_kws_human_campaign_v1,v1.0.0-human,Proprietary-Consented-Research-Commercial,CONSENT_H_S_a1d3f5_v1.0,SPK_H_S_a1d3f5,Southern,MALE,31-50,UNKNOWN,4,Bên trái có ngã tư,carrier_sentence_embedded,SES_H_a1d3f5_01,Galaxy S22,builtin_mobile,living_room_reverberant,desk_distance_60cm,normal,1.85,-21.30,HUMAN_CONSENTED,data/raw/human_recordings_v1/audio_raw/SPK_H_S_a1d3f5/SES_H_a1d3f5_01/rec_h_unknown_SPK_H_S_a1d3f5_SES_H_01_take015.wav,data/raw/human_recordings_v1/audio_16k_standardized/UNKNOWN/rec_h_unknown_SPK_H_S_a1d3f5_SES_H_01_take015.wav,5DBDAA59F8CA64020559D219AE385ED2D68F2A1B01C2DFB8F1FD402E689E42D9,FDF16A23F2AC24301BD1B393517F498562FFEF78A520156A3A2E7C30EF7CEE41,APPROVED,
rec_h_silence_SPK_H_C_4e7c09_SES_H_01_take030,vietnamese_kws_human_campaign_v1,v1.0.0-human,Proprietary-Consented-Research-Commercial,CONSENT_H_C_4e7c09_v1.0,SPK_H_C_4e7c09,Central,FEMALE,18-30,SILENCE,5,,background_room_tone,SES_H_4e7c09_01,Redmi Note 11,builtin_mobile,office_hvac,desk_distance_60cm,ambient_silence,1.60,-48.20,HUMAN_CONSENTED,data/raw/human_recordings_v1/audio_raw/SPK_H_C_4e7c09/SES_H_4e7c09_01/rec_h_silence_SPK_H_C_4e7c09_SES_H_01_take030.wav,data/raw/human_recordings_v1/audio_16k_standardized/SILENCE/rec_h_silence_SPK_H_C_4e7c09_SES_H_01_take030.wav,402E689E42D95DBDAA59F8CA64020559D219AE385ED2D68F2A1B01C2DFB8F1FD,23F2AC24301BD1B393517F498562FFEF78A520156A3A2E7C30EF7CEE41FDF16A,APPROVED,
```
