"""《茉莉花》记谱 + 各站编曲 → 多轨 MIDI、骨架音频、旋律单轨、音符数据。
用法： PY scripts/arr.py [站名…]   （不写站名=全部）
产物： <HUANYOU_OUT>/midi/<站>.mid   <HUANYOU_OUT>/skel/<站>.wav（全编曲）  <HUANYOU_OUT>/skel/<站>_mel.wav（只有旋律）  <HUANYOU_OUT>/score/<站>.json
记谱来源：维基百科 Mo Li Hua 条目的 LilyPond 谱（C 调，12 小节，前两小节反复），逐音对过。
"""
import json, math, os, subprocess, sys
from pathlib import Path
import numpy as np, pretty_midi as pm, soundfile as sf

SRC = Path(__file__).parent
HERE = Path(os.environ.get('HUANYOU_OUT') or Path.cwd() / 'huanyou_out')          # 所有产物都写到这里
SF = Path(os.environ.get('HUANYOU_SF') or SRC.parent / 'sf/MuseScore_General.sf3')
SR = 48000
TONIC = 60                      # 1 = C4
BASE_BPM = 92                   # 原版和四个单开关版的速度

# ---------------------------------------------------------------- 记谱
# 每小节：(拍位, 时值, 距主音的半音数)。1=0 2=2 3=4 5=7 6=9 高音1=12 低音6=-3 低音5=-5
W = {
 1: [(0, 1, 4), (1, .5, 4), (1.5, .5, 7), (2, .5, 9), (2.5, .5, 12), (3, .5, 12), (3.5, .5, 9)],   # 好一朵美丽的
 2: [(0, 1, 7), (1, .5, 7), (1.5, .5, 9), (2, 1, 7)],                                              # 茉莉花
 3: [(0, 1, 7), (1, 1, 7), (2, 1, 7), (3, .5, 4), (3.5, .5, 7)],                                   # 芬芳美丽
 4: [(0, 1, 9), (1, 1, 9), (2, 2, 7)],                                                             # 满枝桠
 5: [(0, 1, 4), (1, .5, 2), (1.5, .5, 4), (2, 1, 7), (3, .5, 4), (3.5, .5, 2)],                    # 又香又白
 6: [(0, 1, 0), (1, .5, 0), (1.5, .5, 2), (2, 2, 0)],                                              # 人人夸
 7: [(0, .5, 4), (.5, .5, 2), (1, .5, 0), (1.5, .5, 4), (2, 1.5, 2), (3.5, .5, 4)],                # 让我来将你
 8: [(0, 1, 7), (1, .5, 9), (1.5, .5, 12), (2, 2, 7)],                                             # 摘下
 9: [(0, 1, 2), (1, .5, 4), (1.5, .5, 7), (2, .5, 2), (2.5, .5, 4), (3, .5, 0), (3.5, .5, -3)],    # 送给别人
 10: [(0, 2, -5), (2, 1, -3), (3, 1, 0)],                                                          # 家 茉莉
 11: [(0, 1.5, 2), (1.5, .5, 4), (2, .5, 0), (2.5, .5, 2), (3, .5, 0), (3.5, .5, -3)],             # 花呀茉莉
 12: [(0, 2, -5)],                                                                                 # 花
}
ORDER = [1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]          # 14 小节
NB = len(ORDER)
# 原版和声，每半小节一个
BASE_CH = 'C C|C G|C C|C G|C C|F C|Am G|F C|Am G|C C|G Am|C F|G F|C C'
# 《小星星》前两句（反向彩蛋用）
STAR = [(0, 1, 0), (1, 1, 0), (2, 1, 7), (3, 1, 7), (4, 1, 9), (5, 1, 9), (6, 2, 7),
        (8, 1, 5), (9, 1, 5), (10, 1, 4), (11, 1, 4), (12, 1, 2), (13, 1, 2), (14, 2, 0)]

Q = {'': [0, 4, 7], 'm': [0, 3, 7], '5': [0, 7], 'sus4': [0, 5, 7], '6': [0, 4, 7, 9], '69': [0, 4, 7, 9, 14], 'm6': [0, 3, 7, 9],
     '7': [0, 4, 7, 10], 'maj7': [0, 4, 7, 11], 'm7': [0, 3, 7, 10], 'maj9': [0, 4, 7, 11, 14], 'm9': [0, 3, 7, 10, 14], '9': [0, 4, 7, 10, 14],
     '13': [0, 4, 10, 14, 21], '7b9': [0, 4, 7, 10, 13], '7#9': [0, 4, 10, 15], '7#11': [0, 4, 10, 18], '9sus': [0, 5, 7, 10, 14], 'm7b5': [0, 3, 6, 10]}
PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def chord(sym):
    r = PC[sym[0]]; i = 1
    while i < len(sym) and sym[i] in '#b': r += 1 if sym[i] == '#' else -1; i += 1
    return r % 12, Q[sym[i:]]


def chords(s):
    out = [c for bar in s.split('|') for c in bar.split()]
    assert len(out) == NB * 2, len(out)
    return out


def near(pc, ref):
    """离 ref 最近的、音级为 pc 的音高"""
    p = ref - ((ref - pc) % 12)
    return p if ref - p <= 6 else p + 12


def voicing(sym, center=60, drop_root=False, n=4):
    r, iv = chord(sym)
    tones = [(r + i) % 12 for i in iv]
    if drop_root and len(tones) > 3: tones = tones[1:]
    return sorted({near(t, center) for t in tones[-n:] if True} | ({near(tones[0], center)} if not drop_root else set()))


class Song:
    """拍为单位的音符容器。mel_scale：旋律时值放大倍数（快节奏风格把旋律拉长一倍）"""
    def __init__(self, name, bpm, mel_scale=1, swing=0.5, bpm_end=None, tonic=TONIC):
        self.name, self.bpm, self.ms, self.swing, self.bpm_end, self.tonic = name, bpm, mel_scale, swing, bpm_end, tonic
        self.tr = {}            # 轨名 → dict(prog, drum, vol, notes[])
        self.acc = None         # 加速用多少拍完成（默认整首）
        self.meta = {}

    def track(self, name, prog=0, drum=False, vol=100):
        self.tr.setdefault(name, dict(prog=prog, drum=drum, vol=vol, notes=[])); return name

    def add(self, tr, beat, dur, pitch, vel=90):
        """pitch 可以带小数：62.4 表示比 D 高 40 音分（钢琴上没有的音），渲染时用弯音轮实现"""
        d = self.tr[tr]['drum']
        self.tr[tr]['notes'].append((float(beat), float(dur), int(pitch) if d else round(pitch + self.tonic - TONIC, 2), int(max(1, min(127, vel)))))

    @property
    def beats(self): return NB * 4 * self.ms

    def sw(self, b):
        """摇摆：把每拍的后半拍往后推"""
        if self.swing == 0.5: return b
        i, f = math.floor(b), b - math.floor(b); s = self.swing
        return i + (f * 2 * s if f < 0.5 else s + (f - 0.5) * 2 * (1 - s))

    def sec(self, b):
        b = self.sw(b)
        if not self.bpm_end: return b * 60 / self.bpm
        # 线性加速：瞬时速度从 bpm 到 bpm_end
        T = self.acc or self.beats; a, c = self.bpm, (self.bpm_end - self.bpm) / T
        return 60 / c * math.log((a + c * min(b, T)) / a) + max(0, b - T) * 60 / self.bpm_end

    def midi(self, only=None):
        m = pm.PrettyMIDI(initial_tempo=self.bpm)
        for name, t in self.tr.items():
            if only and name not in only: continue
            groups = {}                      # 同一轨里微分音不同的音分到不同通道，各自设弯音
            for b, d, p, v in t['notes']:
                fr = 0 if t['drum'] else round(p - round(p), 2)
                groups.setdefault(fr, []).append((b, d, int(round(p)), v))
            for fr, notes in groups.items():
                ins = pm.Instrument(program=t['prog'], is_drum=t['drum'], name=name)
                ins.control_changes.append(pm.ControlChange(7, t['vol'], 0))
                if fr: ins.pitch_bends.append(pm.PitchBend(int(fr / 2 * 8191), 0))
                g = t.get('glide')           # 滑音：每个音从低 g 个半音滑上来（单旋律轨才用）
                for b, d, p, v in notes:
                    st, e = self.sec(b), self.sec(b + d)
                    ins.notes.append(pm.Note(velocity=v, pitch=p, start=st, end=max(e, st + 0.03)))
                    if g and d >= t.get('glide_min', 0.5):
                        for k in range(7): ins.pitch_bends.append(pm.PitchBend(int(-g / 2 * 8191 * (1 - k / 6) ** 2), st + 0.11 * k / 6))
                m.instruments.append(ins)
        return m


# ---------------------------------------------------------------- 旋律
def melody(smap=None, scale=1, bars=None):
    """→ [(拍, 时值, 音高)]。smap：音级重映射 {原半音级: 新半音级}，只挪音阶、不动走向"""
    out = []
    for n, w in enumerate(ORDER):
        if bars is not None and n not in bars: continue
        for pos, d, semi in W[w]:
            pc = semi % 12; new = (smap or {}).get(pc, pc)
            out.append(((n * 4 + pos) * scale, d * scale, TONIC + semi + (new - pc)))
    return out


def put_melody(s, tr, notes, vel=104, oct=0, legato=0.95, accent=True):
    for b, d, p in notes:
        v = vel + (6 if accent and abs(b / s.ms % 4) < 1e-6 else 0) + (3 if accent and abs(b / s.ms % 2) < 1e-6 else 0)
        s.add(tr, b, d * legato, p + 12 * oct, v)


def grace(s, tr, notes, step_fn, vel=80, every=lambda b, d: d >= 1, oct=0, gl=0.12):
    """倚音：在长音前面加一个很短的邻音"""
    for b, d, p in notes:
        if every(b, d): s.add(tr, b - gl, gl, step_fn(p) + 12 * oct, vel)


def trill(s, tr, b, d, p, upper, rate=8, vel=84):
    n = max(2, int(d * rate))
    for i in range(n): s.add(tr, b + i * d / n, d / n * 0.95, upper if i % 2 else p, vel - (0 if i == 0 else 8))


# ---------------------------------------------------------------- 各站
def hb(s):                      # 每半小节的起拍
    return [(i, i * 2 * s.ms) for i in range(NB * 2)]


def base_kit(s, ch, mel, mel_prog=107, comp_prog=46, perc=True):
    """原版编曲：筝旋律 + 竖琴分解和弦 + 低音根音 + 木鱼。第 1–4 站的"只动一个开关"都在它上面改"""
    s.track('mel', mel_prog, vol=118); put_melody(s, 'mel', mel)
    s.track('comp', comp_prog, vol=82); s.track('bass', 32, vol=92)
    for i, b in hb(s):
        r, iv = chord(ch[i]); v = voicing(ch[i], 55, n=3)
        for k, p in enumerate((v + v[::-1])[:4]): s.add('comp', b + k * 0.5, 0.9, p, 62 - k * 3)
        s.add('bass', b, 1.9, near(r, 41), 86)
    if perc:
        s.track('perc', 0, drum=True, vol=74)
        for bt in range(int(s.beats)): s.add('perc', bt, .2, 77 if bt % 2 == 0 else 76, 70 if bt % 4 == 0 else 52)


def orig():
    s = Song('00_orig', BASE_BPM); base_kit(s, chords(BASE_CH), melody(), perc=False)
    s.track('fl', 73, vol=84)                                        # 笛子：反复时高八度跟一遍
    for b, d, p in melody(bars={2, 3, 9, 13}): s.add('fl', b, d * .95, p + 12, 74)
    s.meta = dict(cn='原版', scale=[0, 2, 4, 7, 9], moved=0)
    return s


def base():                      # 实验底版：原版加上最朴素的打点，供单开关对比
    s = Song('00_base', BASE_BPM); base_kit(s, chords(BASE_CH), melody())
    s.meta = dict(cn='底版', scale=[0, 2, 4, 7, 9])
    return s


JP = {2: 1, 4: 5, 9: 8, 11: 0}            # 都节：1 2 3 5 6 → 1 ♭2 4 5 ♭6


def jp_a():                      # 只动音阶：编曲原样，所有音过一遍都节音阶
    s = Song('01_japan_a', BASE_BPM); mel = melody(JP)
    s.track('mel', 107, vol=118); put_melody(s, 'mel', mel)
    s.track('comp', 46, vol=82); s.track('bass', 32, vol=92); s.track('perc', 0, drum=True, vol=74)
    for i, b in hb(s):
        r, iv = chord(chords(BASE_CH)[i]); tones = sorted({near(JP.get((r + x) % 12, (r + x) % 12), 55) for x in iv})
        for k, p in enumerate((tones + tones[::-1])[:4]): s.add('comp', b + k * 0.5, 0.9, p, 62 - k * 3)
        s.add('bass', b, 1.9, near(JP.get(r, r), 41), 86)
    for bt in range(int(s.beats)): s.add('perc', bt, .2, 77 if bt % 2 == 0 else 76, 70 if bt % 4 == 0 else 52)
    s.meta = dict(cn='日本·只动音阶', scale=[0, 1, 5, 7, 8], moved=3)
    return s


def jp_b():                      # 整套：慢、留白、筝的倚音和刮奏、尺八长音、太鼓
    s = Song('01_japan', 66); mel = melody(JP); sc = [0, 1, 5, 7, 8]
    s.track('mel', 107, vol=120); put_melody(s, 'mel', mel, vel=100)
    up = lambda p: min(q for q in range(p + 1, p + 7) if (q - TONIC) % 12 in sc)
    grace(s, 'mel', mel, up, every=lambda b, d: d >= 1 and b % 4 == 0)
    s.track('shaku', 77, vol=94); s.track('koto2', 107, vol=78); s.track('taiko', 116, vol=110); s.track('bell', 112, vol=60); s.track('sham', 106, vol=96)
    for b, d, p in mel:
        if d >= 1 and b % 2 == 1: s.add('sham', b + .5, .3, p - 12, 84)                    # 三味线在弱拍后半拍拨一下
    for n in range(NB):
        b = n * 4
        s.add('shaku', b, 3.6, TONIC + (7 if n % 2 == 0 else 0), 58)                       # 尺八空五度长音
        for k, x in enumerate([-12, -5, 0, 1, 5][: 5]): s.add('koto2', b + 2 + k * .12, 1.6, TONIC - 12 + x + 12 * (k > 2), 56)   # 琴的分解
        if n % 2 == 1:
            run = [TONIC + 24 + x for x in (8, 7, 5, 1, 0)] + [TONIC + 12 + x for x in (8, 7, 5, 1, 0)]   # 句尾下行刮奏
            for k, p in enumerate(run): s.add('koto2', b + 3 + k * .09, .5, p, 60 - k * 2)
        s.add('taiko', b, .8, 43, 96 if n % 2 == 0 else 70)
        if n % 4 == 3: s.add('taiko', b + 3, .4, 43, 78); s.add('taiko', b + 3.5, .4, 43, 88)
        if n % 4 == 0: s.add('bell', b, 2, TONIC + 31, 50)
    s.meta = dict(cn='日本', scale=sc, moved=3)
    return s


BOSSA_CLAVE = [0, 1.5, 3, 5, 6.5]         # 两小节一循环


def syncopate(mel, ms=1):
    """波萨诺瓦式切分：落在第 1、3 拍的音提前半拍（前面有空位才提）"""
    out = []; mel = sorted(mel)
    for i, (b, d, p) in enumerate(mel):
        pos = b % 2
        if pos == 0 and b > 0 and i > 0:
            pb, pd, pp = out[-1]
            if pd >= 1.0: out[-1] = (pb, pd - .5, pp); b, d = b - .5, d + .5
        out.append((b, d, p))
    return out


def bossa_a():                   # 只动节奏：音和乐器原样，旋律切分 + 打点换成波萨诺瓦
    s = Song('02_bossa_a', BASE_BPM); ch = chords(BASE_CH); mel = syncopate(melody())
    s.track('mel', 107, vol=118); put_melody(s, 'mel', mel, accent=False)
    s.track('comp', 46, vol=82); s.track('bass', 32, vol=92); s.track('perc', 0, drum=True, vol=78)
    for i, b in hb(s):
        r, iv = chord(ch[i]); v = voicing(ch[i], 55, n=3)
        for off in ((0, 1.5) if i % 2 == 0 else (.5, 1.5)): [s.add('comp', b + off, .45, p, 60) for p in v]
        s.add('bass', b, 1.4, near(r, 41), 86); s.add('bass', b + 1.5, .5, near((r + 7) % 12, 41), 74)
    for n in range(0, NB, 2):
        for c in BOSSA_CLAVE: s.add('perc', n * 4 + c, .2, 76, 78)
    for e in range(int(s.beats * 2)): s.add('perc', e * .5, .1, 70, 46 if e % 2 else 38)
    s.meta = dict(cn='波萨诺瓦·只动节奏', scale=[0, 2, 4, 7, 9], moved=0)
    return s


BOSSA_CH = 'Cmaj9 C69|Cmaj9 G13|Cmaj9 C69|Dm9 G13|Cmaj9 C69|Fmaj9 C69|Am9 G13|Fmaj9 C69|Am9 G13|Cmaj9 C69|G13 Am9|Cmaj9 Fmaj9|G13 Fmaj9|Cmaj9 C69'


def bossa_b():
    s = Song('02_bossa', 122); ch = chords(BOSSA_CH); mel = syncopate(melody())
    s.track('mel', 73, vol=112); put_melody(s, 'mel', mel, vel=96, oct=1, accent=False)
    s.track('vib', 11, vol=70); put_melody(s, 'vib', [m for m in mel if m[0] >= 8], vel=60, accent=False)
    s.track('gtr', 24, vol=96); s.track('bass', 32, vol=100); s.track('dr', 0, drum=True, vol=86)
    pat = [(0, .9), (1.5, .4), (2.5, .4), (3, .9), (4.5, .4), (5.5, .4), (6.5, .9)]          # 吉他：两小节的波萨扫弦型
    for n in range(0, NB, 2):
        for off, d in pat:
            b = n * 4 + off; c = ch[min(int((b + .5) // 2), NB * 2 - 1)]
            for p in voicing(c, 57, drop_root=True, n=4): s.add('gtr', b, d, p, 66)
    for i, b in hb(s):
        r, _ = chord(ch[i]); s.add('bass', b, 1.4, near(r, 40), 90); s.add('bass', b + 1.5, .5, near((r + 7) % 12, 40), 76)
    for n in range(0, NB, 2):
        for c in BOSSA_CLAVE: s.add('dr', n * 4 + c, .2, 37, 74)
    for e in range(int(s.beats * 2)):
        s.add('dr', e * .5, .1, 42, 50 if e % 2 else 40); s.add('dr', e * .5, .1, 70, 34)
        if e % 4 in (0, 3): s.add('dr', e * .5, .2, 36, 70)
    s.meta = dict(cn='波萨诺瓦', scale=[0, 2, 4, 7, 9], moved=0)
    return s


JAZZ_CH = 'Cmaj9 A7b9|Dm9 G13|Cmaj9 A7b9|Dm9 Db7#11|Cmaj9 C9|Fmaj9 Bb9|Em7 A7b9|Am7 D9|Am9 G13|Cmaj9 E7#9|Am7 D9|Dm9 G9sus|Em7 A7b9|Dm9 C69'


def jazz_a():                    # 只动和弦：旋律、节奏、乐器原样，和弦全部加料
    s = Song('03_jazz_a', BASE_BPM); base_kit(s, chords(JAZZ_CH), melody())
    s.tr['comp']['notes'] = []
    for i, b in hb(s):                                               # 分解和弦的节奏型原样，只是每个和弦换成三度、七度、延伸音
        r, iv = chord(chords(JAZZ_CH)[i]); v = sorted(near((r + x) % 12, 57) for x in iv[1:][-3:])
        for k, p in enumerate((v + v[::-1])[:4]): s.add('comp', b + k * 0.5, 0.9, p, 62 - k * 3)
    s.meta = dict(cn='爵士·只动和弦', scale=[0, 2, 4, 7, 9], moved=0)
    return s


def walk(ch, lo=36):
    """走路低音：每拍一个音，根音→和弦音→和弦音→半音靠近下一个根音"""
    out = []
    for i, c in enumerate(ch):
        r, iv = chord(c); nr = chord(ch[(i + 1) % len(ch)])[0]
        a = near(r, lo + 5); b2 = near(nr, a)
        out += [(i * 2, a), (i * 2 + 1, b2 + (1 if b2 < a else -1) if b2 != a else a + iv[1])]
    return out


def jazz_b():
    s = Song('03_jazz', 116, swing=2 / 3); ch = chords(JAZZ_CH); mel = melody()
    s.track('mel', 66, vol=124); put_melody(s, 'mel', mel, vel=106, legato=.92)              # 次中音萨克斯
    grace(s, 'mel', mel, lambda p: p - 1, vel=70, every=lambda b, d: d >= 1 and b % 4 == 0, gl=.1)
    s.track('pno', 0, vol=72); s.track('bass', 32, vol=100); s.track('dr', 0, drum=True, vol=80)
    for i, b in hb(s):                                                                     # 钢琴：查尔斯顿节奏垫和弦
        v = voicing(ch[i], 62, drop_root=True, n=4)
        for off, d in ((0, .4), (1.5, .4)) if i % 2 == 0 else ((.5, .4),): [s.add('pno', b + off, d, p, 62) for p in v]
    for b, p in walk(ch): s.add('bass', b, .95, p, 92)
    for bt in range(int(s.beats)):
        s.add('dr', bt, .3, 51, 72 if bt % 2 == 0 else 80)
        if bt % 2 == 1: s.add('dr', bt + .5, .2, 51, 58); s.add('dr', bt, .1, 44, 66)
        if bt % 8 == 7: s.add('dr', bt + .5, .2, 38, 50)
    s.meta = dict(cn='爵士', scale=[0, 2, 4, 7, 9], moved=0)
    return s


def country_a():                 # 只换乐器：一个音不改
    s = Song('04_country_a', BASE_BPM); base_kit(s, chords(BASE_CH), melody(), mel_prog=105)
    s.meta = dict(cn='乡村·只换乐器', scale=[0, 2, 4, 7, 9], moved=0)
    return s


def country_b():
    s = Song('04_country', 108); ch = chords(BASE_CH); mel = melody()
    s.track('mel', 110, vol=106); put_melody(s, 'mel', mel, vel=92)                         # 提琴唱旋律
    grace(s, 'mel', mel, lambda p: p - 2, vel=72, every=lambda b, d: d >= 1, gl=.1)
    s.track('banjo', 105, vol=112); s.track('gtr', 25, vol=80); s.track('bass', 32, vol=100); s.track('dr', 0, drum=True, vol=80)
    at = {round(b * 4): p for b, d, p in mel}
    for q in range(int(s.beats * 4)):                                                      # 班卓琴滚奏：旋律音嵌在十六分音符里，高音弦一直响 5
        b = q / 4; c = ch[int(b // 2)]; r, iv = chord(c)
        p = at.get(q) or [near((r + iv[1]) % 12, 62), TONIC + 19, near((r + iv[2]) % 12, 64), TONIC + 19][q % 4]
        s.add('banjo', b, .3, p, 96 if q in at else 62)
    for i, b in hb(s):
        r, iv = chord(ch[i]); v = voicing(ch[i], 57, n=3)
        s.add('bass', b, .9, near(r, 38), 92); s.add('bass', b + 1, .9, near((r + 7) % 12, 38), 84)
        for off in (.5, 1.5): [s.add('gtr', b + off, .4, p, 64) for p in v]
    for e in range(int(s.beats * 4)):
        s.add('dr', e * .25, .1, 38 if e % 8 == 4 else 42, 76 if e % 8 == 4 else (44 if e % 2 else 56))
        if e % 8 == 0: s.add('dr', e * .25, .2, 36, 78)
    s.meta = dict(cn='乡村', scale=[0, 2, 4, 7, 9], moved=0)
    return s


def irish():                     # 四拍 → 六八拍吉格：每拍分成"长短"，长音加回音
    s = Song('05_irish', 118, swing=2 / 3); mel = melody(); sc = [0, 2, 4, 7, 9]
    ch = chords('C C|C G|C C|C G|C Bb|F C|Am G|F C|Am G|C Bb|G Am|C F|G F|C C')
    step = lambda p, k: sorted(q for q in range(p - 9, p + 10) if (q - TONIC) % 12 in sc and (q > p) == (k > 0) and q != p)[0 if k > 0 else -1]
    jig = []
    for b, d, p in mel:
        if d == 1: jig += [(b, .5, p), (b + .5, .5, step(p, 1) if round(b) % 2 else p)]
        elif d >= 2: jig += [(b, 1, p), (b + 1, .5, step(p, 1)), (b + 1.5, d - 1.5, p)]
        else: jig.append((b, d, p))
    s.track('mel', 72, vol=108); put_melody(s, 'mel', jig, vel=92, oct=1, legato=.85)       # 短笛当哨笛
    s.track('fid', 110, vol=86); put_melody(s, 'fid', jig, vel=74, legato=.9)
    s.track('gtr', 25, vol=84); s.track('bass', 32, vol=92); s.track('bod', 0, drum=True, vol=96)
    for i, b in hb(s):
        r, iv = chord(ch[i]); v = voicing(ch[i], 55, n=3)
        s.add('bass', b, 1.9, near(r, 38), 86)
        for off, vel in ((0, 70), (.5, 48), (1, 62), (1.5, 48)): [s.add('gtr', b + off, .4, p, vel) for p in v]
    for bt in range(int(s.beats)):
        s.add('bod', bt, .3, 45 if bt % 2 == 0 else 47, 86 if bt % 2 == 0 else 70); s.add('bod', bt + .5, .2, 47, 54)
        s.add('bod', bt + .25, .1, 47, 40)
    s.meta = dict(cn='爱尔兰', scale=sc, moved=0, meter='6/8')
    return s


ME = {2: 1, 9: 8}                # 希贾兹骨架：1 ♭2 3 5 ♭6（♭2 到 3 是增二度）


def mideast():
    s = Song('06_mideast', 96); mel = melody(ME); sc = [0, 1, 4, 5, 7, 8, 10]
    s.track('mel', 40, vol=112); s.track('oud', 24, vol=112); s.track('str', 48, vol=92); s.track('ney', 75, vol=70)
    s.track('drone', 42, vol=74); s.track('dr', 0, drum=True, vol=112)
    up = lambda p: min(q for q in range(p + 1, p + 5) if (q - TONIC) % 12 in sc)
    for b, d, p in mel:
        if d >= 1.5: trill(s, 'mel', b, d, p, up(p), rate=6, vel=96)                        # 长音上自动打颤
        elif d >= 1: s.add('mel', b, .25, p, 100); s.add('mel', b + .25, .25, up(p), 84); s.add('mel', b + .5, d - .55, p, 96)
        else: s.add('mel', b, d * .95, p, 98)
        s.add('str', b, d * .95, p + 12, 74); s.add('str', b, d * .95, p, 70)               # 弦乐高低八度齐奏
        n = max(1, int(d * 4))                                                              # 乌德琴：同音轮指
        for k in range(n): s.add('oud', b + k * d / n, d / n * .9, p - 12, 88 - (k % 2) * 14)
    for n in range(NB):
        b = n * 4; s.add('drone', b, 3.9, TONIC - 24, 70); s.add('drone', b, 3.9, TONIC - 17, 58)
        if n % 2 == 1: [s.add('ney', b + 2.5 + k * .25, .24, TONIC + 12 + x, 62) for k, x in enumerate((8, 7, 5, 4, 1, 0))]   # 句尾的下行过门
        for off, note, vel in ((0, 61, 112), (.5, 60, 92), (1.5, 60, 88), (2, 61, 104), (3, 60, 94), (3.5, 54, 70), (1, 54, 52), (2.5, 54, 52), (.25, 54, 40), (3.25, 60, 60)):   # 马克苏姆节奏：咚 哒 – 哒 咚 – 哒 –
            s.add('dr', b + off, .2, note, vel)
    s.meta = dict(cn='中东', scale=[0, 1, 4, 7, 8], moved=2)
    return s


def blues():                     # 6 换成 ♭7，3 前面蹭一个 ♭3，三连音摇摆，全属七和弦
    s = Song('07_blues', 82, swing=2 / 3); mel = melody({9: 10})
    ch = chords('C7 C7|C7 G7|C7 C7|C7 C7|C7 C7|F7 C7|C7 G7|F7 C7|C7 G7|C7 C7|G7 F7|C7 F7|G7 F7|C7 C7')
    s.track('mel', 22, vol=127); put_melody(s, 'mel', mel, vel=114, legato=.95); s.track('mel8', 27, vol=96); put_melody(s, 'mel8', mel, vel=84, oct=-1, legato=.9)
    for b, d, p in mel:
        if (p - TONIC) % 12 == 4: s.add('mel', b - .14, .14, p - 1, 84)                    # 蓝调三度
    s.track('gtr', 27, vol=66); s.track('org', 16, vol=44); s.track('bass', 33, vol=100); s.track('dr', 0, drum=True, vol=80)
    for i, b in hb(s):
        r, iv = chord(ch[i]); root = near(r, 43)
        for k in range(4):                                                                   # 吉他：五度—六度的布基律动
            s.add('gtr', b + k * .5, .45, root, 70); s.add('gtr', b + k * .5, .45, root + (7 if k // 1 % 2 == 0 else 9), 66)
        [s.add('org', b, 1.9, p, 46) for p in voicing(ch[i], 60, drop_root=True, n=3)]
        s.add('bass', b, .9, near(r, 36), 94); s.add('bass', b + 1, .9, near((r + (4 if i % 2 == 0 else 7)) % 12, 38), 84)
    for bt in range(int(s.beats)):
        s.add('dr', bt, .3, 42, 62); s.add('dr', bt + .5, .2, 42, 46)
        s.add('dr', bt, .3, 36 if bt % 2 == 0 else 38, 84 if bt % 2 == 0 else 80)
    s.meta = dict(cn='蓝调', scale=[0, 2, 4, 7, 10], moved=1)
    return s


MIN = {4: 3, 9: 8, 11: 10}       # 小调：3、6 各降半音


def russian():                   # 小调 + 手风琴 + 嘣嚓嘣嚓 + 越拉越快
    s = Song('08_russian', 84, bpm_end=184); s.acc = 16; mel = melody(MIN)
    ch = chords('Cm Cm|Cm G7|Cm Cm|Cm G7|Cm Cm|Fm Cm|Cm G7|Fm Cm|Cm G7|Cm Cm|G7 Cm|Cm Fm|G7 Fm|Cm Cm')
    s.track('mel', 21, vol=127); put_melody(s, 'mel', mel, vel=110); put_melody(s, 'mel', mel, vel=88, oct=1)
    s.track('bal', 25, vol=92); s.track('acc', 21, vol=54); s.track('tuba', 58, vol=94); s.track('dr', 0, drum=True, vol=80)
    for b, d, p in mel:                                                                      # 三角琴：高八度轮指
        n = max(1, int(d * 4))
        for k in range(n): s.add('bal', b + k * d / n, d / n * .8, p + 12, 70 - (k % 2) * 12)
    for i, b in hb(s):
        r, iv = chord(ch[i]); v = voicing(ch[i], 58, n=3)
        s.add('tuba', b, .8, near(r, 38), 96); s.add('tuba', b + 1, .8, near((r + 7) % 12, 38), 88)
        for off in (.5, 1.5): [s.add('acc', b + off, .35, p, 70) for p in v]
        for off in (.5, 1.5): s.add('dr', b + off, .1, 54, 70)
        s.add('dr', b, .2, 36, 80); s.add('dr', b + 1, .2, 36, 72)
    s.add('dr', s.beats - 2, 1, 49, 100); [s.add('acc', s.beats - 2, 1.5, p, 96) for p in voicing('Cm', 60)]
    s.meta = dict(cn='俄罗斯', scale=[0, 2, 3, 7, 8], moved=2)
    return s


def baroque():                   # 羽管键琴两声部：右手旋律加颤音，左手八分音符跑动，低一个八度晚两拍的卡农
    s = Song('09_baroque', 92); mel = melody(); sc = [0, 2, 4, 5, 7, 9, 11]
    ch = chords('C C|C G|C Am|D7 G|C C7|F C|Am E7|F C|Am D7|G C|G Am|C F|G7 F|C C')
    s.track('mel', 6, vol=116); s.track('canon', 6, vol=84); s.track('lh', 6, vol=92); s.track('vc', 42, vol=84)
    up = lambda p: min(q for q in range(p + 1, p + 4) if (q - TONIC) % 12 in sc)
    for b, d, p in mel:
        if d >= 2: trill(s, 'mel', b, 1.5, p, up(p), rate=8, vel=92); s.add('mel', b + 1.5, d - 1.55, p, 90)
        elif d >= 1 and b % 4 == 0: [s.add('mel', b + k * .125, .12, q, 92) for k, q in enumerate((p, up(p)))]; s.add('mel', b + .25, d - .3, p, 96)   # 波音
        else: s.add('mel', b, d * .92, p, 96)
    for b, d, p in mel:
        if b + 2 < s.beats - 4: s.add('canon', b + 2, d * .9, p - 12, 76)
    for i, b in hb(s):
        r, iv = chord(ch[i]); nr = chord(ch[(i + 1) % len(ch)])[0]; a = near(r, 45)
        line = [a, near((r + iv[1]) % 12, a + 3), near((r + iv[2]) % 12, a + 5), near(nr, a + 4) + (1 if near(nr, a + 4) < a + 4 else -1)]
        for k, p in enumerate(line): s.add('lh', b + k * .5, .45, p, 76)
        s.add('vc', b, 1.9, near(r, 38), 76)
    s.meta = dict(cn='巴洛克', scale=[0, 2, 4, 7, 9], moved=0)
    return s


def chip():                      # 8-bit：四个声道，和弦拆成飞快的琶音。这一站不用音色库，用 chip.py 合成
    s = Song('10_8bit', 150); ch = chords(BASE_CH); mel = melody()
    s.track('mel', 80, vol=110); put_melody(s, 'mel', mel, vel=100, oct=1, legato=.85)
    s.track('arp', 80, vol=70); s.track('tri', 38, vol=100); s.track('noise', 0, drum=True, vol=80)
    for i, b in hb(s):
        r, iv = chord(ch[i]); v = voicing(ch[i], 64, n=3)
        for k in range(16): s.add('arp', b + k * .125, .12, (v + [v[0] + 12])[k % 4], 60)
        for k in range(4): s.add('tri', b + k * .5, .4, near(r, 36) + (12 if k % 2 else 0), 90)
    for e in range(int(s.beats * 2)):
        s.add('noise', e * .5, .1, 42, 50)
        if e % 4 == 0: s.add('noise', e * .5, .2, 36, 90)
        if e % 4 == 2: s.add('noise', e * .5, .2, 38, 84)
    s.meta = dict(cn='8-bit', scale=[0, 2, 4, 7, 9], moved=0, synth='chip')
    return s


def anime():                     # 王道进行 4536 + 八分音符猛推 + 第二遍主句升一个全音
    s = Song('11_anime', 172, mel_scale=2); mel = melody(scale=2)
    prog = ['F', 'G', 'Em', 'Am']; ch = [prog[i % 4] for i in range(NB * 2)]; ch[-2:] = ['F', 'G']
    LIFT = 2 * 8                                                                            # 第二遍主句起升一个全音
    tr = lambda b: 2 if b >= LIFT else 0
    s.track('mel', 61, vol=112); s.track('lead', 29, vol=96); s.track('strm', 48, vol=96); s.track('gtr', 30, vol=108); s.track('str', 48, vol=70)
    s.track('pno', 0, vol=78); s.track('bass', 34, vol=110); s.track('dr', 0, drum=True, vol=112)
    for b, d, p in mel:
        s.add('mel', b, d * .92, p + 12 + tr(b), 104); s.add('lead', b, d * .95, p + tr(b), 92); s.add('strm', b, d * .95, p + 24 + tr(b), 84)
    for i in range(NB * 2):
        b = i * 4; r, iv = chord(ch[i]); t = tr(b); root = near(r, 40) + t
        for k in range(8):
            v = 96 if k % 2 == 0 else 84
            s.add('gtr', b + k * .5, .45, root, v); s.add('gtr', b + k * .5, .45, root + 7, v - 4); s.add('gtr', b + k * .5, .45, root + 12, v - 8)
            s.add('bass', b + k * .5, .45, root - 12, 96)
        for p in voicing(ch[i], 67, n=3): s.add('str', b, 3.9, p + t, 62)
        for k, p in enumerate(voicing(ch[i], 72, n=3) * 3): s.add('pno', b + k * .5, .4, p + t, 64) if k < 8 else None
        for k in range(8):
            s.add('dr', b + k * .5, .1, 42, 80 if k % 2 == 0 else 60)
            if k in (0, 3, 5): s.add('dr', b + k * .5, .2, 36, 112)
            if k in (2, 6): s.add('dr', b + k * .5, .2, 38, 116)
        if i % 4 == 0: s.add('dr', b, 1, 49, 108)
        if i % 4 == 3 or b + 4 == LIFT: [s.add('dr', b + 2 + k * .25, .1, (38, 38, 50, 50, 47, 47, 45, 45)[k], 88 + k * 4) for k in range(8)]
    s.meta = dict(cn='动漫片头', scale=[0, 2, 4, 7, 9], moved=0, lift=2)
    return s


PHR = {2: 1, 4: 3, 9: 8, 11: 10}  # 弗里几亚：2、3、6 各降半音


def metal():                     # 三个音压低 + 强力和弦 + 闷音十六分 + 双踩
    s = Song('12_metal', 176, mel_scale=2); mel = melody(PHR, scale=2)
    ch = chords('C5 C5|C5 Ab5|C5 C5|C5 G5|C5 C5|Ab5 C5|Eb5 G5|Ab5 C5|Eb5 G5|C5 C5|G5 Ab5|C5 Db5|G5 Ab5|C5 C5')
    s.track('mel', 30, vol=127); s.track('mel2', 29, vol=78); s.track('rh', 30, vol=78); s.track('bass', 34, vol=100); s.track('dr', 0, drum=True, vol=96)
    sc = [0, 1, 3, 5, 7, 8, 10]
    third = lambda p: max(q for q in range(p - 5, p - 2) if (q - TONIC) % 12 in sc)
    for b, d, p in mel: s.add('mel', b, d * .96, p + 12, 112); s.add('mel', b, d * .96, p + 24, 92); s.add('mel2', b, d * .96, third(p) + 12, 76)
    for i in range(NB * 2):
        b = i * 4; r, iv = chord(ch[i]); root = near(r, 38)
        for k in range(16):
            acc = k in (0, 6, 12); p = root if acc else TONIC - 24
            s.add('rh', b + k * .25, .22, p, 100 if acc else 74)
            if acc: s.add('rh', b + k * .25, .22, p + 7, 96); s.add('rh', b + k * .25, .22, p + 12, 90)
            s.add('bass', b + k * .25, .22, p - 12 if p - 12 >= 28 else p, 96)
            s.add('dr', b + k * .25, .1, 36, 104 if k % 2 == 0 else 92)
        s.add('dr', b + 2, .3, 38, 118)
        for k in range(4): s.add('dr', b + k, .2, 42 if (i * 4 + k) % 8 else 49, 84)
    s.meta = dict(cn='重金属', scale=[0, 1, 3, 7, 8], moved=3)
    return s


def square():                    # 广场舞：动次打次 + 反拍低音 + 锯齿主音
    s = Song('13_square', 134); ch = chords(BASE_CH); mel = melody()
    s.track('mel', 81, vol=104); put_melody(s, 'mel', mel, vel=98, oct=1, legato=.8)
    s.track('suona', 111, vol=96); put_melody(s, 'suona', mel, vel=90, legato=.9)
    s.track('stab', 62, vol=80); s.track('bass', 38, vol=108); s.track('dr', 0, drum=True, vol=104)
    for i, b in hb(s):
        r, iv = chord(ch[i]); v = voicing(ch[i], 62, n=3)
        for k in range(2):
            s.add('dr', b + k, .2, 36, 116); s.add('dr', b + k + .5, .15, 46, 86); s.add('bass', b + k + .5, .4, near(r, 38), 104)
            [s.add('stab', b + k + .5, .3, p, 76) for p in v]
        s.add('dr', b + 1, .2, 39, 100)
    s.meta = dict(cn='广场舞', scale=[0, 2, 4, 7, 9], moved=0)
    return s


def mash():                      # 不存在的国家：都节音阶 + 波萨诺瓦节奏 + 失真吉他
    s = Song('14_mash', 122); mel = syncopate(melody(JP))
    s.track('mel', 30, vol=124); put_melody(s, 'mel', mel, vel=108, accent=False)
    s.track('gtr', 24, vol=76); s.track('bass', 32, vol=100); s.track('dr', 0, drum=True, vol=90); s.track('koto', 107, vol=84)
    for n in range(0, NB, 2):
        for off, d in [(0, .9), (1.5, .4), (2.5, .4), (3, .9), (4.5, .4), (5.5, .4), (6.5, .9)]:
            for p in (TONIC - 12, TONIC - 7, TONIC - 5 + 6, TONIC + 1): s.add('gtr', n * 4 + off, d, p, 64)
        for c in BOSSA_CLAVE: s.add('dr', n * 4 + c, .2, 37, 78)
    for i, b in hb(s): s.add('bass', b, 1.4, TONIC - 24, 90); s.add('bass', b + 1.5, .5, TONIC - 17, 76)
    for e in range(int(s.beats * 2)): s.add('dr', e * .5, .1, 42, 50 if e % 2 else 40)
    for n in range(1, NB, 2): [s.add('koto', n * 4 + 3 + k * .09, .5, TONIC + 12 + x, 62) for k, x in enumerate((8, 7, 5, 1, 0))]
    s.meta = dict(cn='不存在的国家', scale=[0, 1, 5, 7, 8], moved=3)
    return s


def star():                      # 反向彩蛋：《小星星》入境中国。4 换成 5（五声里没有 4），筝加倚音和刮奏，笛子加花
    s = Song('15_star_cn', 88); sc = [0, 2, 4, 7, 9]
    s.track('mel', 107, vol=120); s.track('fl', 73, vol=90); s.track('comp', 46, vol=80); s.track('bass', 32, vol=88); s.track('perc', 0, drum=True, vol=70)
    mel = [(b, d, TONIC + {5: 7}.get(x, x)) for b, d, x in STAR]
    put_melody(s, 'mel', mel, vel=102)
    up = lambda p: min(q for q in range(p + 1, p + 5) if (q - TONIC) % 12 in sc)
    dn = lambda p: max(q for q in range(p - 4, p) if (q - TONIC) % 12 in sc)
    grace(s, 'mel', mel, dn, every=lambda b, d: b % 2 == 0, gl=.1)
    for b, d, p in mel:                                                                      # 笛子加花：每个长一点的音后面带一个上邻音
        s.add('fl', b, d * .5, p + 12, 76); s.add('fl', b + d * .5, d * .25, up(p + 12), 66); s.add('fl', b + d * .75, d * .2, p + 12, 66)
    for i, c in enumerate('C C C F C G C C'.split()):
        b = i * 2; r, iv = chord(c); v = [near(x, 55) for x in sorted({(r + q) % 12 for q in iv} & set(sc) | {r})]
        for k, p in enumerate((v + v[::-1])[:4]): s.add('comp', b + k * .5, .9, p, 60)
        s.add('bass', b, 1.9, near(r, 41), 80)
    for k, x in enumerate([24, 21, 19, 16, 14, 12, 9, 7, 4, 2, 0]): s.add('comp', 7 + k * .08, .6, TONIC + x, 64 - k)   # 句中刮奏
    for bt in range(16): s.add('perc', bt, .2, 77 if bt % 2 == 0 else 76, 60)
    s.add('perc', 14, 2, 52, 70)
    s._beats = 16
    s.meta = dict(cn='小星星·中国风', scale=sc, moved=2)
    return s


STYLES = dict(orig=orig, base=base, japan_a=jp_a, japan=jp_b, bossa_a=bossa_a, bossa=bossa_b, jazz_a=jazz_a, jazz=jazz_b, country_a=country_a,
              country=country_b, irish=irish, mideast=mideast, blues=blues, russian=russian, baroque=baroque, chip=chip, anime=anime, metal=metal,
              square=square, mash=mash, star=star)


# ---------------------------------------------------------------- 渲染
def fluid(mid, wav):
    subprocess.run(['fluidsynth', '-ni', '-g', '0.7', '-r', str(SR), '-F', str(wav), str(SF), str(mid)], check=True, capture_output=True)


def norm(path, peak=0.89, dur=None):
    """裁掉合成器拖出来的长尾巴（留 1.5 秒余音并淡出），再把峰值拉齐"""
    a, sr = sf.read(path, always_2d=True)
    if dur: a = a[: int((dur + 1.5) * sr)]; f = int(0.6 * sr); a[-f:] *= np.linspace(1, 0, f)[:, None]
    a = a / (np.abs(a).max() + 1e-9) * peak; sf.write(path, a.astype(np.float32), sr)


def export(s):
    for d in ('midi', 'skel', 'score'): (HERE / d).mkdir(parents=True, exist_ok=True)
    mid = HERE / f'midi/{s.name}.mid'; s.midi().write(str(mid))
    melmid = HERE / f'midi/{s.name}_mel.mid'; s.midi(only={'mel'}).write(str(melmid))
    full, melw = HERE / f'skel/{s.name}.wav', HERE / f'skel/{s.name}_mel.wav'
    if s.meta.get('synth') == 'chip':
        import chip as chipsyn
        chipsyn.render(s, full, SR); chipsyn.render(s, melw, SR, only={'mel'})
    else:
        fluid(mid, full); fluid(melmid, melw)
    beats = getattr(s, '_beats', s.beats)
    norm(full, dur=s.sec(beats)); norm(melw, dur=s.sec(beats))
    seen = set(); notes = []                 # 同一拍上叠了八度的，只留最低的那个音，画面上的旋律线才不会乱跳
    for b, d, p, v in sorted(s.tr['mel']['notes'], key=lambda n: (n[0], n[2])):
        if round(b, 4) in seen: continue
        seen.add(round(b, 4)); notes.append(dict(t=round(s.sec(b), 4), d=round(s.sec(b + d) - s.sec(b), 4), p=p, beat=b))
    bars = [round(s.sec(i * 4 * s.ms), 4) for i in range(int(beats / (4 * s.ms)) + 1)]
    json.dump(dict(name=s.name, bpm=s.bpm, bpm_end=s.bpm_end, swing=s.swing, mel_scale=s.ms, dur=round(s.sec(beats), 3), bars=bars, notes=notes,
                   tracks={k: dict(prog=v['prog'], drum=v['drum'], n=len(v['notes'])) for k, v in s.tr.items()}, **s.meta),
              open(HERE / f'score/{s.name}.json', 'w'), ensure_ascii=False, indent=1)
    print(f'{s.name:16s} {s.meta.get("cn", ""):10s} bpm {s.bpm}{"→" + str(s.bpm_end) if s.bpm_end else "":5s} 时长 {s.sec(beats):5.1f}s 轨 {len(s.tr)} 音符 {sum(len(t["notes"]) for t in s.tr.values())}')


RELAY = ['japan', 'bossa', 'jazz', 'country', 'irish', 'mideast', 'blues', 'russian', 'baroque', 'chip', 'anime', 'metal']


def relay(bpm=104):
    """接龙：前 6 小节（好一朵美丽的茉莉花 ×2 + 芬芳美丽满枝桠），每半小节换一站，十二站轮一遍。
    每站都按同一个速度重新渲染整段，再在半小节线上硬切拼起来。"""
    import chip as chipsyn
    slot = 2 * 60 / bpm; n = int(slot * SR); out = []; notes = []; tmp = HERE / 'skel/_relay_tmp'
    tmp.mkdir(exist_ok=True)
    for k, name in enumerate(RELAY):
        s = STYLES[name](); s.bpm, s.bpm_end = bpm * s.ms, None
        wav = tmp / f'{name}.wav'
        if s.meta.get('synth') == 'chip': chipsyn.render(s, wav, SR)
        else: mid = tmp / f'{name}.mid'; s.midi().write(str(mid)); fluid(mid, wav)
        a, _ = sf.read(wav, always_2d=True); i0 = k * n
        seg = a[i0:i0 + n + (int(1.2 * SR) if k == len(RELAY) - 1 else 0)].copy()
        seg = seg / (np.sqrt(np.mean(seg ** 2)) + 1e-9) * 0.16                              # 各站拉到同一响度
        f = 240; seg[:f] *= np.linspace(0, 1, f)[:, None]; seg[-f:] *= np.linspace(1, 0, f)[:, None]
        if k == len(RELAY) - 1: g = int(1.0 * SR); seg[-g:] *= np.linspace(1, 0, g)[:, None]
        out.append(seg)
        lo, hi = k * 2 * s.ms, (k + 1) * 2 * s.ms
        seen = set()
        for b, d, p, v in sorted(s.tr['mel']['notes'], key=lambda n: (n[0], n[2])):
            if round(b, 4) in seen: continue
            seen.add(round(b, 4))
            if lo <= b < hi and d >= 0.2 * s.ms / 2: notes.append(dict(t=round(s.sec(b), 4), d=round(min(s.sec(b + d), (k + 1) * slot) - s.sec(b), 4), p=p, style=s.name))
    a = np.concatenate(out); a = a / (np.abs(a).max() + 1e-9) * 0.89
    sf.write(HERE / 'skel/16_relay.wav', a.astype(np.float32), SR)
    json.dump(dict(name='16_relay', cn='接龙', bpm=bpm, dur=round(len(RELAY) * slot, 3), slot=round(slot, 4), order=RELAY, notes=notes,
                   bars=[round(i * 2 * slot, 4) for i in range(7)]), open(HERE / 'score/16_relay.json', 'w'), ensure_ascii=False, indent=1)
    print(f'16_relay         接龙         每站 {slot:.2f}s 共 {len(RELAY) * slot:.1f}s 旋律音 {len(notes)}')


if __name__ == '__main__':
    sys.path.insert(0, str(SRC))
    for k in sys.argv[1:] or list(STYLES) + ['relay']:
        relay() if k == 'relay' else export(STYLES[k]())
