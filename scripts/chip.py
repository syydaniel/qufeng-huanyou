"""8-bit 合成器：两个方波声道、一个三角波声道、一个噪声声道，照着红白机的做法。给 arr.py 的 chip 站用。"""
import numpy as np, soundfile as sf


def hz(p): return 440.0 * 2 ** ((p - 69) / 12)


def pulse(f, n, sr, duty=0.25, vib=0.0):
    t = np.arange(n) / sr
    ph = np.cumsum(f * (1 + vib * 0.006 * np.sin(2 * np.pi * 6 * t) * np.clip((t - 0.18) * 6, 0, 1))) / sr
    return np.where(ph % 1 < duty, 1.0, -1.0)


def tri(f, n, sr):
    ph = (np.arange(n) * f / sr) % 1
    w = 4 * np.abs(ph - 0.5) - 1
    return np.round(w * 7.5) / 7.5                    # 16 级台阶，红白机三角波的颗粒感


def noise(n, sr, kind):
    rng = np.random.default_rng(7)
    hold = {36: 24, 38: 6, 42: 2}.get(kind, 4)        # 噪声采样保持越长，听起来越低
    x = np.repeat(rng.uniform(-1, 1, n // hold + 1), hold)[:n]
    dec = {36: 0.09, 38: 0.11, 42: 0.03}.get(kind, 0.05)
    env = np.exp(-np.arange(n) / sr / dec)
    if kind == 36: x = x * 0.6 + np.sign(np.sin(2 * np.pi * np.cumsum(120 * np.exp(-np.arange(n) / sr / 0.04) + 45) / sr)) * 0.8
    return x * env


def render(song, path, sr=48000, only=None):
    dur = song.sec(song.beats) + 1.0
    out = np.zeros(int(dur * sr))
    gain = {'mel': 0.30, 'arp': 0.13, 'tri': 0.34, 'noise': 0.22}
    for name, t in song.tr.items():
        if only and name not in only: continue
        for b, d, p, v in t['notes']:
            i0 = int(song.sec(b) * sr); n = max(64, int((song.sec(b + d) - song.sec(b)) * sr))
            if name == 'noise': w = noise(max(n, int(0.2 * sr)), sr, p)
            elif name == 'tri': w = tri(hz(p), n, sr)
            else:
                w = pulse(hz(p), n, sr, duty=0.5 if name == 'mel' else 0.125, vib=1.0 if name == 'mel' else 0.0)
                w *= np.clip(1.0 - 0.35 * np.arange(n) / n, 0, 1)        # 音量台阶式衰减
            f = min(len(w), 96); w[:f] *= np.linspace(0, 1, f); w[-f:] *= np.linspace(1, 0, f)
            w = w * gain.get(name, 0.2) * v / 100
            j = min(len(out), i0 + len(w)); out[i0:j] += w[: j - i0]
    out = np.stack([out, out], 1)
    sf.write(path, (out / (np.abs(out).max() + 1e-9) * 0.89).astype(np.float32), sr)
