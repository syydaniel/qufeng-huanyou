"""风格像不像：用 CLAP（laion/larger_clap_general，文字-音频对照模型，第一次用会自动下载）给每段音频打分。
它替我"听"：把音频和各站的文字描述放进同一个空间，看这段音频离自己那一站的描述近不近、排第几。
用法： $ACESTEP_ROOT/.venv/bin/python scripts/style.py <音频…>     → 写 <HUANYOU_OUT>/style_scores.json
注意：这是模型的判断，不是人耳，只能当筛选用。
"""
import json, os, re, sys
if os.environ.get('HUANYOU_OFFLINE') == '1': os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'   # 模型已下载过、想离线跑时设 HUANYOU_OFFLINE=1
import numpy as np, soundfile as sf, torch, torchaudio
from transformers import ClapModel, ClapProcessor
HERE = os.environ.get('HUANYOU_OUT') or os.path.join(os.getcwd(), 'huanyou_out')
TXT = {
 '00': 'traditional Chinese folk music played on guzheng and bamboo flute',
 '01': 'traditional Japanese music played on koto and shakuhachi',
 '02': 'Indonesian gamelan orchestra with bronze metallophones and gongs',
 '03': 'Indian classical raga with sitar, tanpura drone and tabla',
 '04': 'Arabic Middle Eastern music with oud, violins and darbuka',
 '05': 'Russian folk dance with accordion and balalaika',
 '06': 'Spanish flamenco with guitar, hand claps and castanets',
 '07': 'Irish jig with tin whistle, fiddle and bodhran',
 '08': 'slow shuffle blues with harmonica and electric guitar',
 '09': 'bossa nova with nylon guitar',
 '10': 'swing jazz with saxophone and piano',
 '11': 'heavy metal with distorted guitars',
 '12': '8-bit chiptune video game music',
 '13': 'baroque harpsichord music',
}
m = ClapModel.from_pretrained('laion/larger_clap_general').eval(); pr = ClapProcessor.from_pretrained('laion/larger_clap_general')
keys = list(TXT)
with torch.no_grad():
    t = m.get_text_features(**pr(text=[TXT[k] for k in keys], return_tensors='pt', padding=True)); t = torch.nn.functional.normalize(t, dim=-1)
out = os.path.join(HERE, 'style_scores.json'); db = json.load(open(out)) if os.path.exists(out) else {}
for f in sys.argv[1:]:
    a, sr = sf.read(f, always_2d=True); a = torch.tensor(a.mean(1), dtype=torch.float32)
    if sr != 48000: a = torchaudio.functional.resample(a, sr, 48000)
    a = a[: 48000 * 30]; win = 48000 * 10
    chunks = [a[i:i + win].numpy() for i in range(0, max(1, len(a) - win + 1), win // 2)] or [a.numpy()]
    with torch.no_grad():
        e = m.get_audio_features(**pr(audios=chunks, sampling_rate=48000, return_tensors='pt')); e = torch.nn.functional.normalize(e, dim=-1).mean(0); e = e / e.norm()
    sim = (t @ e).numpy(); order = np.argsort(-sim); mm = re.search(r'(\d\d)_', os.path.basename(f)); own = mm.group(1) if mm else ''
    r = dict(own=round(float(sim[keys.index(own)]), 3) if own in keys else None, rank=int(list(order).index(keys.index(own))) + 1 if own in keys else None,
             top=[[keys[i], round(float(sim[i]), 3)] for i in order[:3]])
    db[os.path.relpath(f, HERE)] = r
    print(f"{os.path.basename(f):44s} 本站 {r['own']}  排第 {r['rank']}  前三 {r['top']}", flush=True)
json.dump(db, open(out, 'w'), ensure_ascii=False, indent=1)
