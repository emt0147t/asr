# VieNeu-TTS v3 Turbo — Voice Roster Verification Report

**Workspace:** `D:\trainidentificationaudio`  
**Target Directory:** `experiments/synthetic_kws_tts_pilot_vieneu_local/`  
**Installed SDK:** `vieneu==3.8.3` in `.venv_vieneu` (Python 3.11.9)  
**Upstream Metadata Source:** `vieneu/assets/voices_v3_turbo.json`  
**Model Weights Checkpoint:** `pnnbao-ump/VieNeu-TTS-v3-Turbo` (subfolder: `onnx_update`)  
**Associated Data File:** [`voice_roster_verification.json`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_tts_pilot_vieneu_local/voice_roster_verification.json)  
**Date:** October 17, 2026  
**Auditor Verdict:** `VERIFIED_CONCORDANT`

---

## 1. Upstream Preset Voice Roster (`tts.list_preset_voices()`)

The authoritative method `tts.list_preset_voices()` was executed directly on the installed `vieneu` engine in `.venv_vieneu`. Exactly 25 preset voices are returned. Editors' picks (marked with `featured: 1..10` in the upstream JSON) appear first with a `⭐` prefix, followed by standard presets.

| # | Preset Name (`voice_id`) | Display Label | Gender | Region | Upstream Style Token | Upstream Description | Featured Rank | Aliases |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :-: | :--- |
| 1 | `Adam bựa` | ⭐ Adam bựa — Nam · Bắc · Phong cách tự nhiên | `male` | `Bắc` | `tu_nhien` | Nam · Bắc · Phong cách tự nhiên | 1 | None |
| 2 | `Trúc Ly` | ⭐ Trúc Ly — Nữ · Bắc · Phong cách tự nhiên | `female` | `Bắc` | `tu_nhien` | Nữ · Bắc · Phong cách tự nhiên | 2 | None |
| 3 | `Thiện Minh` | ⭐ Thiện Minh — Nam · Bắc · Phong cách kể chuyện | `male` | `Bắc` | `ke_chuyen` | Nam · Bắc · Phong cách kể chuyện | 3 | `['Anh Khôi']` |
| 4 | `Mai Anh` | ⭐ Mai Anh — Nữ · Bắc · Phong cách tin tức | `female` | `Bắc` | `tin_tuc` | Nữ · Bắc · Phong cách tin tức | 4 | None |
| 5 | `Hải Đăng` | ⭐ Hải Đăng — Nam · Bắc · Phong cách tự nhiên | `male` | `Bắc` | `tu_nhien` | Nam · Bắc · Phong cách tự nhiên | 5 | `['Minh Quân Pro', 'Minh Quân']` |
| 6 | `Thùy Dung` | ⭐ Thùy Dung — Nữ · Nam · Phong cách tin tức | `female` | `Nam` | `tin_tuc` | Nữ · Nam · Phong cách tin tức | 6 | None |
| 7 | `Thiền Tâm Đức` | ⭐ Thiền Tâm Đức — Nam · Bắc · Phong cách kể chuyện | `male` | `Bắc` | `ke_chuyen` | Nam · Bắc · Phong cách kể chuyện | 7 | None |
| 8 | `Ngọc Huyền` | ⭐ Ngọc Huyền — Nữ · Bắc · Giọng đọc tự nhiên | `female` | `Bắc` | `tu_nhien` | Nữ · Bắc · Giọng đọc tự nhiên | 8 | None |
| 9 | `Quang Sơn` | ⭐ Quang Sơn — Nam · Trung · Phong cách tự nhiên | `male` | `Trung` | `tu_nhien` | Nam · Trung · Phong cách tự nhiên | 9 | None |
| 10 | `Ngọc Trân` | ⭐ Ngọc Trân — Nữ · Trung · Phong cách tự nhiên | `female` | `Trung` | `tu_nhien` | Nữ · Trung · Phong cách tự nhiên | 10 | None |
| 11 | `Minh Đức` | Minh Đức — Nam · Bắc · Phong cách tin tức | `male` | `Bắc` | `tin_tuc` | Nam · Bắc · Phong cách tin tức | None | None |
| 12 | `Phạm Tuyên` | Phạm Tuyên — Nam · Bắc · Phong cách tự nhiên | `male` | `Bắc` | `tu_nhien` | Nam · Bắc · Phong cách tự nhiên | None | None |
| 13 | `Thái Sơn` | Thái Sơn — Nam · Nam · Phong cách kể chuyện | `male` | `Nam` | `ke_chuyen` | Nam · Nam · Phong cách kể chuyện | None | None |
| 14 | `Xuân Vĩnh` | Xuân Vĩnh — Nam · Bắc · Phong cách tự nhiên | `male` | `Bắc` | `tu_nhien` | Nam · Bắc · Phong cách tự nhiên | None | None |
| 15 | `Thanh Bình` | Thanh Bình — Nam · Bắc · Phong cách kể chuyện | `male` | `Bắc` | `ke_chuyen` | Nam · Bắc · Phong cách kể chuyện | None | None |
| 16 | `Ngọc Linh` | Ngọc Linh — Nữ · Bắc · Phong cách kể chuyện | `female` | `Bắc` | `ke_chuyen` | Nữ · Bắc · Phong cách kể chuyện | None | None |
| 17 | `Đoan Trang` | Đoan Trang — Nữ · Bắc · Phong cách tự nhiên | `female` | `Bắc` | `tu_nhien` | Nữ · Bắc · Phong cách tự nhiên | None | None |
| 18 | `Thục Đoan` | Thục Đoan — Nữ · Nam · Phong cách kể chuyện | `female` | `Nam` | `ke_chuyen` | Nữ · Nam · Phong cách kể chuyện | None | None |
| 19 | `Minh Triết` | Minh Triết — Nam · Nam · Phong cách tin tức | `male` | `Nam` | `tin_tuc` | Nam · Nam · Phong cách tin tức | None | None |
| 20 | `Mỹ Duyên` | Mỹ Duyên — Nữ · Nam · Phong cách đọc truyện | `female` | `Nam` | `ke_chuyen` | Nữ · Nam · Phong cách đọc truyện | None | None |
| 21 | `Quỳnh Anh` | Quỳnh Anh — Nữ · Bắc · Phong cách đọc truyện | `female` | `Bắc` | `ke_chuyen` | Nữ · Bắc · Phong cách đọc truyện | None | None |
| 22 | `Đức Trí` | Đức Trí — Nam · Nam · Phong cách đọc truyện | `male` | `Nam` | `ke_chuyen` | Nam · Nam · Phong cách đọc truyện | None | None |
| 23 | `Kim Thanh` | Kim Thanh — Nữ · Nam · Phong cách đọc truyện | `female` | `Nam` | `ke_chuyen` | Nữ · Nam · Phong cách đọc truyện | None | None |
| 24 | `Adam` | Adam — Nam · Nam · Giọng đọc tự nhiên | `male` | `Nam` | `tu_nhien` | Nam · Nam · Giọng đọc tự nhiên | None | None |
| 25 | `Quốc Tuấn` | Quốc Tuấn — Nam · Bắc · Phong cách tự nhiên | `male` | `Bắc` | `tu_nhien` | Nam · Bắc · Phong cách tự nhiên | None | `['Mạnh Dũng']` |

---

## 2. Investigation of `Hải Đăng` and Default Voice Status

### 2.1 The Upstream Default Voice Transition
Upstream documentation on Hugging Face and GitHub documents the following version evolution:
- **VieNeu SDK v3.3.0:** 20 preset voices; default voice was `Adam`.
- **VieNeu SDK v3.7.1:** 23 preset voices; default voice was documented as `Minh Quân`.
- **VieNeu SDK v3.8.3 (Currently Installed):** 25 preset voices; default voice was formally changed to `Hải Đăng`.

### 2.2 Forensic Inspection of Installed Code and Assets
1. **JSON Declaration:** In `vieneu/assets/voices_v3_turbo.json` (line 4):
   ```json
   "default_voice": "Hải Đăng"
   ```
2. **SDK Ingestion:** In `vieneu/v3turbo.py` (lines 142–143):
   ```python
   self._default_voice = data.get("default_voice")
   logger.info(f"📢 Loaded {len(self._preset_voices)} preset voices (default: {self._default_voice})")
   ```
3. **Runtime Engine Property:** Running `tts._default_voice` in `.venv_vieneu` returns `'Hải Đăng'`. Calling `tts.infer("text")` without specifying `voice` uses `'Hải Đăng'`.
4. **Alias Resolution:** The voice previously known as `Minh Quân` or `Minh Quân Pro` was renamed to `Hải Đăng`. In `voices_v3_turbo.json`, the entry for `Hải Đăng` contains:
   ```json
   "aliases": ["Minh Quân Pro", "Minh Quân"]
   ```
   When a caller requests `voice="Minh Quân"`, `v3turbo.py` resolves the alias directly to `Hải Đăng`.
5. **Conclusion:** The statement that `Hải Đăng` is the default voice in the installed version (`3.8.3`) is **100% verified by codebase and asset evidence**.

---

## 3. Pilot Manifest Voices vs Upstream Metadata

Every voice utilized in [`vieneu_local_pilot_manifest.csv`](file:///d:/trainidentificationaudio/experiments/synthetic_kws_tts_pilot_vieneu_local/vieneu_local_pilot_manifest.csv) was audited against the upstream `voices_v3_turbo.json` asset.

| Manifest Voice ID | Manifest Gender | Upstream Gender | Gender Status | Manifest Region | Upstream Region (VI / EN) | Region Status | Manifest Style | Upstream Description |
| :--- | :--- | :--- | :-: | :--- | :--- | :-: | :--- | :--- |
| **Mai Anh** | `female` | `female` | **MATCH** | `North` | `Bắc` / North | **MATCH** | `Tin tức` | Nữ · Bắc · Phong cách tin tức |
| **Trúc Ly** | `female` | `female` | **MATCH** | `North` | `Bắc` / North | **MATCH** | `Tự nhiên` | Nữ · Bắc · Phong cách tự nhiên |
| **Ngọc Trân** | `female` | `female` | **MATCH** | `Central` | `Trung` / Central | **MATCH** | `Tự nhiên` | Nữ · Trung · Phong cách tự nhiên |
| **Thùy Dung** | `female` | `female` | **MATCH** | `South` | `Nam` / South | **MATCH** | `Tin tức` | Nữ · Nam · Phong cách tin tức |
| **Hải Đăng** | `male` | `male` | **MATCH** | `North` | `Bắc` / North | **MATCH** | `Tự nhiên` | Nam · Bắc · Phong cách tự nhiên |
| **Phạm Tuyên** | `male` | `male` | **MATCH** | `North` | `Bắc` / North | **MATCH** | `Tự nhiên` | Nam · Bắc · Phong cách tự nhiên |
| **Quang Sơn** | `male` | `male` | **MATCH** | `Central` | `Trung` / Central | **MATCH** | `Tự nhiên` | Nam · Trung · Phong cách tự nhiên |
| **Thái Sơn** | `male` | `male` | **MATCH** | `South` | `Nam` / South | **MATCH** | `Kể chuyện` | Nam · Nam · Phong cách kể chuyện |

* **Zero Name Inferences:** Gender and dialect regional metadata were **strictly extracted** from the upstream metadata keys `gender` and `region` in `voices_v3_turbo.json`. No linguistic or cultural name assumptions were made.
* **Match Rate:** **8 / 8 (100.0%)** exact correspondence between manifest fields and upstream metadata.

---

## 4. Confirmation of Voice-to-WAV Assignment

All 32 generated clips in `experiments/synthetic_kws_tts_pilot_vieneu_local/` have exact, trace-verified voice assignments confirmed by generation logs, manifest records, and file naming conventions:

| # | Clip ID | Target Keyword | Voice ID (`voice_id`) | Gender | Region | KWS WAV File Path | SHA-256 (First 16 chars) |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `vieneu_pilot_trai_maianh_f_north` | `TRAI` | `Mai Anh` | female | North | `.../vieneu_pilot_trai_maianh_f_north_kws_16k.wav` | `c2a11400d0c3ebf5` |
| 2 | `vieneu_pilot_trai_trucly_f_north` | `TRAI` | `Trúc Ly` | female | North | `.../vieneu_pilot_trai_trucly_f_north_kws_16k.wav` | `5c6130423c4aebfa` |
| 3 | `vieneu_pilot_trai_ngoctran_f_central` | `TRAI` | `Ngọc Trân` | female | Central | `.../vieneu_pilot_trai_ngoctran_f_central_kws_16k.wav` | `df33e38719bc4ddc` |
| 4 | `vieneu_pilot_trai_thuydung_f_south` | `TRAI` | `Thùy Dung` | female | South | `.../vieneu_pilot_trai_thuydung_f_south_kws_16k.wav` | `9d5598150ea2feef` |
| 5 | `vieneu_pilot_trai_haidang_m_north` | `TRAI` | `Hải Đăng` | male | North | `.../vieneu_pilot_trai_haidang_m_north_kws_16k.wav` | `fa7cfb8fbfd88583` |
| 6 | `vieneu_pilot_trai_phamtuyen_m_north` | `TRAI` | `Phạm Tuyên` | male | North | `.../vieneu_pilot_trai_phamtuyen_m_north_kws_16k.wav` | `be4523eb45c6dc85` |
| 7 | `vieneu_pilot_trai_quangson_m_central` | `TRAI` | `Quang Sơn` | male | Central | `.../vieneu_pilot_trai_quangson_m_central_kws_16k.wav` | `83df8d487cb1a039` |
| 8 | `vieneu_pilot_trai_thaison_m_south` | `TRAI` | `Thái Sơn` | male | South | `.../vieneu_pilot_trai_thaison_m_south_kws_16k.wav` | `e2a4bebaef80b6a6` |
| 9 | `vieneu_pilot_phai_maianh_f_north` | `PHAI` | `Mai Anh` | female | North | `.../vieneu_pilot_phai_maianh_f_north_kws_16k.wav` | `c37b03698ea457cb` |
| 10 | `vieneu_pilot_phai_trucly_f_north` | `PHAI` | `Trúc Ly` | female | North | `.../vieneu_pilot_phai_trucly_f_north_kws_16k.wav` | `bb5609ee356bf488` |
| 11 | `vieneu_pilot_phai_ngoctran_f_central` | `PHAI` | `Ngọc Trân` | female | Central | `.../vieneu_pilot_phai_ngoctran_f_central_kws_16k.wav` | `87fa2bb3b02868ff` |
| 12 | `vieneu_pilot_phai_thuydung_f_south` | `PHAI` | `Thùy Dung` | female | South | `.../vieneu_pilot_phai_thuydung_f_south_kws_16k.wav` | `e5518b5b7c02b376` |
| 13 | `vieneu_pilot_phai_haidang_m_north` | `PHAI` | `Hải Đăng` | male | North | `.../vieneu_pilot_phai_haidang_m_north_kws_16k.wav` | `10243be1211e06ff` |
| 14 | `vieneu_pilot_phai_phamtuyen_m_north` | `PHAI` | `Phạm Tuyên` | male | North | `.../vieneu_pilot_phai_phamtuyen_m_north_kws_16k.wav` | `6c1285223c6f6631` |
| 15 | `vieneu_pilot_phai_quangson_m_central` | `PHAI` | `Quang Sơn` | male | Central | `.../vieneu_pilot_phai_quangson_m_central_kws_16k.wav` | `369a473ee16a4a15` |
| 16 | `vieneu_pilot_phai_thaison_m_south` | `PHAI` | `Thái Sơn` | male | South | `.../vieneu_pilot_phai_thaison_m_south_kws_16k.wav` | `77ea66b571dd6cf0` |
| 17 | `vieneu_pilot_len_maianh_f_north` | `LEN` | `Mai Anh` | female | North | `.../vieneu_pilot_len_maianh_f_north_kws_16k.wav` | `7bc25091ff60eeb6` |
| 18 | `vieneu_pilot_len_trucly_f_north` | `LEN` | `Trúc Ly` | female | North | `.../vieneu_pilot_len_trucly_f_north_kws_16k.wav` | `6e017ce8d80f682f` |
| 19 | `vieneu_pilot_len_ngoctran_f_central` | `LEN` | `Ngọc Trân` | female | Central | `.../vieneu_pilot_len_ngoctran_f_central_kws_16k.wav` | `9b87df7a92263152` |
| 20 | `vieneu_pilot_len_thuydung_f_south` | `LEN` | `Thùy Dung` | female | South | `.../vieneu_pilot_len_thuydung_f_south_kws_16k.wav` | `1734df4360ca0db8` |
| 21 | `vieneu_pilot_len_haidang_m_north` | `LEN` | `Hải Đăng` | male | North | `.../vieneu_pilot_len_haidang_m_north_kws_16k.wav` | `091ba81a8b5774bb` |
| 22 | `vieneu_pilot_len_phamtuyen_m_north` | `LEN` | `Phạm Tuyên` | male | North | `.../vieneu_pilot_len_phamtuyen_m_north_kws_16k.wav` | `ae06ef7f41f71df9` |
| 23 | `vieneu_pilot_len_quangson_m_central` | `LEN` | `Quang Sơn` | male | Central | `.../vieneu_pilot_len_quangson_m_central_kws_16k.wav` | `17665b1695ae540e` |
| 24 | `vieneu_pilot_len_thaison_m_south` | `LEN` | `Thái Sơn` | male | South | `.../vieneu_pilot_len_thaison_m_south_kws_16k.wav` | `1e23ce752c0dfb57` |
| 25 | `vieneu_pilot_xuong_maianh_f_north` | `XUONG` | `Mai Anh` | female | North | `.../vieneu_pilot_xuong_maianh_f_north_kws_16k.wav` | `43fb0fc8ba7d0b3a` |
| 26 | `vieneu_pilot_xuong_trucly_f_north` | `XUONG` | `Trúc Ly` | female | North | `.../vieneu_pilot_xuong_trucly_f_north_kws_16k.wav` | `6b2d26f74a8a5bca` |
| 27 | `vieneu_pilot_xuong_ngoctran_f_central` | `XUONG` | `Ngọc Trân` | female | Central | `.../vieneu_pilot_xuong_ngoctran_f_central_kws_16k.wav` | `fc256a47a165b6ea` |
| 28 | `vieneu_pilot_xuong_thuydung_f_south` | `XUONG` | `Thùy Dung` | female | South | `.../vieneu_pilot_xuong_thuydung_f_south_kws_16k.wav` | `141fe73ff91fb34d` |
| 29 | `vieneu_pilot_xuong_haidang_m_north` | `XUONG` | `Hải Đăng` | male | North | `.../vieneu_pilot_xuong_haidang_m_north_kws_16k.wav` | `2bc168ec1712a321` |
| 30 | `vieneu_pilot_xuong_phamtuyen_m_north` | `XUONG` | `Phạm Tuyên` | male | North | `.../vieneu_pilot_xuong_phamtuyen_m_north_kws_16k.wav` | `6e80b27b3fa99b4f` |
| 31 | `vieneu_pilot_xuong_quangson_m_central` | `XUONG` | `Quang Sơn` | male | Central | `.../vieneu_pilot_xuong_quangson_m_central_kws_16k.wav` | `375d80d287bb24f4` |
| 32 | `vieneu_pilot_xuong_thaison_m_south` | `XUONG` | `Thái Sơn` | male | South | `.../vieneu_pilot_xuong_thaison_m_south_kws_16k.wav` | `173ee0db1d6368d4` |

---

## 5. Scientific Qualifications Regarding Preset Voices

1. **Preset Voices are Neural Conditioning Vectors, NOT Human Participants:**
   * In VieNeu-TTS v3 Turbo, each preset voice consists of:
     - A 192-dimensional speaker embedding vector $\mathbf{e}_{\text{spk}} \in \mathbb{R}^{192}$.
     - A reference audio code matrix $\mathbf{C}_{\text{ref}} \in \mathbb{Z}^{16 \times 68}$ pre-quantized by MOSS-Audio-Tokenizer-Nano.
   * Synthesizing speech with different preset voices conditions the neural decoder on different points in the speaker latent manifold.
   * **Scientific boundary:** These presets provide synthetic timbre and acoustic diversity. They **do not represent independent human participants** in an acoustic survey or clinical field acquisition.
2. **Subjective Intelligibility Freeze:**
   * Objective metrics (spectral centroid, RMS, F0 slope, zero clipping, zero truncation) confirm acoustic signal health.
   * However, perceptual intelligibility must not be inferred automatically from TTS model output.
   * The status remains strictly **`PENDING_HUMAN_LISTENER_VERIFICATION`**.
