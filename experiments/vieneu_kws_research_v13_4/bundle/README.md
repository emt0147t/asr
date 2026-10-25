# Vietnamese Keyword Spotting (KWS) CNN Release Candidate v11

**Status:** `FROZEN_CNN_RELEASE_CANDIDATE`  
**Candidate Checkpoint SHA-256:** `3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2`  
**ONNX Model SHA-256:** `54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99`  
**Operating Threshold:** $\theta^*_{\text{cmd}} = 0.20$

---

## 1. Quickstart & Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run PyTorch inference on a single WAV file
python infer.py path/to/sample.wav --backend pytorch

# 3. Run ONNX Runtime inference on a single WAV file
python infer.py path/to/sample.wav --backend onnx

# 4. Run batch inference on a directory of WAV files
python infer.py --dir path/to/wav_dir --backend onnx --out predictions.json

# 5. Run CPU latency benchmark
python benchmark_latency.py
```

---

## 2. Bundle Contents

```
bundle/
├── checkpoint/
│   └── variant_v11_hard_negative_weighted_seed42.pt   # Frozen PyTorch model checkpoint
├── onnx/
│   ├── vietnamese_kws_v11.onnx                        # Standard FP32 ONNX model (opset 17)
│   └── export_parity_manifest.json                    # ONNX vs PyTorch parity verification
├── infer.py                                           # Standalone inference CLI (PyTorch & ONNX)
├── benchmark_latency.py                               # Latency benchmarking script
├── model_config.json                                  # Architecture, layers, and class metadata
├── preprocessing_contract.json                        # Preprocessing & STFT specification
├── requirements.txt                                   # Python dependencies
├── environment_lock.json                              # Environment and dependency version lock
├── checksums.json                                     # SHA-256 checksums for all bundle files
├── MODEL_CARD.md                                      # Detailed model card
└── README.md                                          # This documentation
```

---

## 3. Audio Input Requirements

- **Sampling Rate:** 16,000 Hz.
- **Channels:** 1 (mono). Multi-channel audio is automatically downmixed to mono.
- **Duration:** 1.000 second (16,000 samples). Shorter clips are right-padded with zeros; longer clips are truncated.
- **Amplitude Normalization:** Clamped to $[-1.0, 1.0]$ only if $\max(|x|) > 1.0$. No floor amplification.
