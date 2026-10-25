import sys
import io
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from experiments.vieneu_kws_research_v13_4.bundle.infer import VietnameseKWSInferencer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

inferencer = VietnameseKWSInferencer(backend='onnx', theta_cmd=0.20)

v12_df = pd.read_csv('experiments/vieneu_kws_research_v12/manifests/dataset_v12_manifest.csv')
conv_df = v12_df[v12_df['reject_subgroup'] == 'conversation']

detections = []
for idx, r in conv_df.iterrows():
    p = Path(r['audio_path'])
    res = inferencer.predict(p)
    if res['predicted_class'] != 'REJECT':
        detections.append({
            'sample_id': r['sample_id'],
            'split': r['split'],
            'predicted_class': res['predicted_class'],
            'confidence': res['confidence'],
            'probabilities': res['probabilities']
        })

print(f"Total v12 conversation clips triggering command at theta=0.20 on v11: {len(detections)} / {len(conv_df)}")
for d in detections[:10]:
    print(f"  {d['sample_id']} ({d['split']}) -> {d['predicted_class']} (conf: {d['confidence']:.3f})")
