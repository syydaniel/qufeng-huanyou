"""客观自检：我听不到声音，用程序量"旋律还在不在、开关到没到位"。
用法： PY scripts/check.py skel            # 查全部骨架
       PY scripts/check.py cover [文件…]   # 查上色版，写 <HUANYOU_OUT>/cover/scores.json
指标：
  hit   旋律命中率：每个旋律音响的那一段里，它的音级是不是当时最响的前 3 个音级之一（中高音区）
  ctrl  对照组：把旋律音高顺序错开再量一遍。hit 要明显高于 ctrl 才算"旋律还在"
  same  音色和骨架有多像（0–1），太接近 1 说明没上到色
  warp  上色版和骨架对齐后，时间上最多偏了多少秒（偏太多画面就卡不上）
  tempo 检测到的速度 / 设定速度
  lufs  响度
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, librosa, pyloudnorm, soundfile as sf
warnings.filterwarnings('ignore')
import os
HERE = Path(os.environ.get('HUANYOU_OUT') or Path.cwd() / 'huanyou_out')
SR, HOP = 22050, 512
FMIN = librosa.midi_to_hz(55)          # 从 G3 往上四个八度，尽量避开低音和鼓


def load(p):
    y, _ = librosa.load(str(p), sr=SR, mono=True); return y


def chroma(y, melody_band=True):
    h = librosa.effects.harmonic(y, margin=2.0)
    if melody_band: c = librosa.feature.chroma_cqt(y=h, sr=SR, hop_length=HOP, fmin=FMIN, n_octaves=4, bins_per_octave=36)
    else: c = librosa.feature.chroma_cqt(y=h, sr=SR, hop_length=HOP, bins_per_octave=36)
    return c / (c.sum(0, keepdims=True) + 1e-9)


def hit_rate(c, notes, tmap=lambda t: t, top=3, shift=0):
    """notes: [{t,d,p}]。shift≠0 时把音高序列错位，当对照组"""
    ps = [n['p'] for n in notes]; ps = ps[shift:] + ps[:shift]
    ok = tot = 0
    for n, p in zip(notes, ps):
        if n['d'] < 0.12: continue                       # 倚音太短不算
        a, b = tmap(n['t']), tmap(n['t'] + n['d'])
        i0, i1 = int(a * SR / HOP), max(int(a * SR / HOP) + 1, int(b * SR / HOP))
        seg = c[:, i0:i1]
        if seg.shape[1] == 0: continue
        e = seg.mean(1); rank = int((e > e[int(round(p)) % 12]).sum())
        ok += rank < top; tot += 1
    return ok / max(1, tot)


def ctrl_rate(c, notes, tmap=lambda t: t, top=3):
    return float(np.mean([hit_rate(c, notes, tmap, top, s) for s in (3, 5, 7, 11, 13)]))


def align(cov, skel, dur):
    """上色版对骨架做动态时间规整 → 骨架时间到上色版时间的映射、最大偏移（只看有音乐的那一段）"""
    n = int((dur + 0.5) * SR); cov, skel = cov[:n], skel[:n]
    A, B = chroma(skel, False), chroma(cov, False)
    D, wp = librosa.sequence.dtw(X=A, Y=B, metric='cosine', global_constraints=True, band_rad=0.08)
    wp = wp[::-1]; ts, tc = wp[:, 0] * HOP / SR, wp[:, 1] * HOP / SR
    uniq = np.unique(ts, return_index=True)[1]
    f = lambda t: float(np.interp(t, ts[uniq], tc[uniq]))
    k = ts < dur
    return f, float(np.abs(ts - tc)[k].max()), float(D[-1, -1] / len(wp))


def same_timbre(cov, skel, dur):
    """音色和骨架有多像（0–1）。太接近 1 说明没上到色，只是把骨架复述了一遍"""
    n = int(dur * SR); a = librosa.feature.mfcc(y=cov[:n], sr=SR, n_mfcc=20, hop_length=HOP)[1:]; b = librosa.feature.mfcc(y=skel[:n], sr=SR, n_mfcc=20, hop_length=HOP)[1:]
    m = min(a.shape[1], b.shape[1]); a, b = a[:, :m], b[:, :m]
    return float(np.mean((a * b).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-9)))


def tempo_of(y):
    return float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP)[0])[0])


def lufs(p):
    a, sr = sf.read(str(p), always_2d=True)
    return float(pyloudnorm.Meter(sr).integrated_loudness(a)), float(np.abs(a).max())


def pc_profile(c):
    return (c.mean(1) / c.mean(1).sum()).round(3).tolist()


def near_ratio(t, target):
    r = t / target
    return min((r, r * 2, r / 2, r * 1.5, r / 1.5), key=lambda x: abs(x - 1))


def check_skel():
    rows = []
    for jp in sorted((HERE / 'score').glob('*.json')):
        sc = json.load(open(jp)); name = sc['name']
        y = load(HERE / f'skel/{name}.wav'); ym = load(HERE / f'skel/{name}_mel.wav')
        c, cm = chroma(y), chroma(ym)
        L, pk = lufs(HERE / f'skel/{name}.wav')
        target = sc['bpm'] if not sc.get('bpm_end') else (sc['bpm'] + sc['bpm_end']) / 2
        rows.append(dict(name=name, cn=sc.get('cn'), stem_hit=round(hit_rate(cm, sc['notes'], top=1), 3), hit=round(hit_rate(c, sc['notes']), 3),
                         ctrl=round(ctrl_rate(c, sc['notes']), 3), tempo=round(near_ratio(tempo_of(y), target), 3), lufs=round(L, 1), peak=round(pk, 2), dur=round(len(y) / SR, 1)))
        r = rows[-1]; print(f"{r['name']:14s} 单轨命中 {r['stem_hit']:.2f}  全编曲命中 {r['hit']:.2f} 对照 {r['ctrl']:.2f}  速度比 {r['tempo']:.2f}  {r['lufs']:6.1f} LUFS 峰值 {r['peak']:.2f}  {r['dur']}s", flush=True)
    json.dump(rows, open(HERE / 'skel/scores.json', 'w'), ensure_ascii=False, indent=1)


def check_cover(files):
    out = HERE / 'cover/scores.json'; db = json.load(open(out)) if out.exists() else {}
    for f in files:
        f = Path(f); name = f.name.split('__')[0]
        sc = json.load(open(HERE / f'score/{name}.json'))
        cov, skel = load(f), load(HERE / f'skel/{name}.wav')
        tmap, warp, cost = align(cov, skel, sc['dur'])
        c = chroma(cov); L, pk = lufs(f)
        target = sc['bpm'] if not sc.get('bpm_end') else (sc['bpm'] + sc['bpm_end']) / 2
        cs = chroma(skel)
        r = dict(hit=round(hit_rate(c, sc['notes'], tmap), 3), ctrl=round(ctrl_rate(c, sc['notes'], tmap), 3), hit1=round(hit_rate(c, sc['notes'], tmap, top=1), 3),
                 skel_hit=round(hit_rate(cs, sc['notes']), 3), same=round(same_timbre(cov, skel, sc['dur']), 3), warp=round(warp, 2), cost=round(cost, 3), tempo=round(near_ratio(tempo_of(cov), target), 3),
                 lufs=round(L, 1), peak=round(pk, 2), dur=round(len(cov) / SR, 1), pcs=pc_profile(c),
                 map=[[round(t, 3), round(tmap(t), 3)] for t in sc['bars']])
        db[f.name] = r
        print(f"{f.name:46s} 命中 {r['hit']:.2f}(骨架 {r['skel_hit']:.2f}) 首位 {r['hit1']:.2f} 对照 {r['ctrl']:.2f}  像骨架 {r['same']:.2f}  偏移 {r['warp']:.2f}s  速度比 {r['tempo']:.2f}  {r['lufs']:.1f} LUFS  {r['dur']}s", flush=True)
        json.dump(db, open(out, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    if sys.argv[1] == 'skel': check_skel()
    else: check_cover(sys.argv[2:] or sorted((HERE / 'cover').glob('*.wav')))
