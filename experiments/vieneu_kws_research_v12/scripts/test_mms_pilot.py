import sys
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from transformers import VitsModel, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

model_id = 'facebook/mms-tts-vie'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = VitsModel.from_pretrained(model_id)
model.eval()

pilot_words = [
    'bật đèn', 'tắt đèn', 'bật quạt', 'tắt quạt',
    'trái', 'phải', 'lên', 'xuống',
    'ma', 'mà', 'má', 'mả', 'mã', 'mạ',
    'luôn', 'làm', 'phim', 'phát', 'trưa', 'trước'
]

out_dir = Path('data/candidate_tts/mms_pilot')
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Testing {len(pilot_words)} words with {model_id}...")
for w in pilot_words:
    inputs = tokenizer(w, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    unk_count = sum(1 for t in tokens if t == '<unk>')
    
    with torch.no_grad():
        output = model(**inputs).waveform[0].numpy()
        
    dur = len(output) / model.config.sampling_rate
    max_amp = float(np.abs(output).max())
    mean_amp = float(np.mean(np.abs(output)))
    
    # Save audio
    safe_name = w.replace(' ', '_')
    wav_path = out_dir / f"{safe_name}.wav"
    sf.write(str(wav_path), output, model.config.sampling_rate)
    
    print(f"  Word: '{w}' -> Dur: {dur:.3f}s, MaxAmp: {max_amp:.3f}, MeanAmp: {mean_amp:.4f}, UNK: {unk_count}, Tokens: {tokens[:10]}")
