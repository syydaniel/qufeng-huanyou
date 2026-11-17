"""曲风环游引擎：给一首五声（或接近五声）的旋律，输出各国版本。每个国家先换音阶，再配上那个国家的节奏、和声和乐器。
用法： python world.py [曲名] [站名…]      或      python world.py --jianpu <英文名> <中文名> "<简谱>" [站名…]
   曲名默认 molihua；曲子写在 TUNES 里，用简谱：1–7 是音级，' 高八度 , 低八度，b/# 升降；
   时值默认一拍，后面加 _ 是半拍、__ 是四分之一拍、. 是附点；- 把前一个音延长一拍；0 是休止；| 分小节。
产物： <HUANYOU_OUT>/skel/w_<曲名>_<站>.wav 等，命名规则和 arr.py 一样。
"""
import json, math, sys
from pathlib import Path
import numpy as np, soundfile as sf
SRC = Path(__file__).parent; sys.path.insert(0, str(SRC))
import arr
HERE = arr.HERE
from arr import Song, chord, near, voicing, trill, TONIC, SR

SEMI = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}
TUNES = {
 'molihua': ('茉莉花', "3 3_ 5_ 6_ 1'_ 1'_ 6_ | 5 5_ 6_ 5 0 | 3 3_ 5_ 6_ 1'_ 1'_ 6_ | 5 5_ 6_ 5 0 | 5 5 5 3_ 5_ | 6 6 5 - | 3 2_ 3_ 5 3_ 2_ | 1 1_ 2_ 1 - | "
                       "3_ 2_ 1_ 3_ 2. 3_ | 5 6_ 1'_ 5 - | 2 3_ 5_ 2_ 3_ 1_ 6,_ | 5, - 6, 1 | 2. 3_ 1_ 2_ 1_ 6,_ | 5, - 0 0"),
 'star': ('小星星', "1 1 5 5 | 6 6 5 - | 4 4 3 3 | 2 2 1 - | 5 5 4 4 | 3 3 2 - | 5 5 4 4 | 3 3 2 - | 1 1 5 5 | 6 6 5 - | 4 4 3 3 | 2 2 1 -"),
 'ode': ('欢乐颂', "3 3 4 5 | 5 4 3 2 | 1 1 2 3 | 3. 2_ 2 - | 3 3 4 5 | 5 4 3 2 | 1 1 2 3 | 2. 1_ 1 -"),
 'tigers': ('两只老虎', "1 2 3 1 | 1 2 3 1 | 3 4 5 - | 3 4 5 - | 5_ 6_ 5_ 4_ 3 1 | 5_ 6_ 5_ 4_ 3 1 | 1 5, 1 - | 1 5, 1 -"),
}


def parse(s):
    """简谱 → [(拍, 时值, 距主音半音数)]，以及小节数"""
    out = []; bars = [b.split() for b in s.split('|') if b.strip()]
    for n, bar in enumerate(bars):
        pos = 0.0
        for tok in bar:
            if tok == '-': b, d, x = out[-1]; out[-1] = (b, d + 1, x); pos += 1; continue
            d = 1.0
            while tok.endswith('_'): tok = tok[:-1]; d /= 2
            if tok.endswith('.'): tok = tok[:-1]; d *= 1.5
            if tok[0] == '0': pos += d; continue
            acc = 0
            while tok[0] in 'b#': acc += -1 if tok[0] == 'b' else 1; tok = tok[1:]
            semi = SEMI[int(tok[0])] + acc + 12 * tok.count("'") - 12 * tok.count(',')
            out.append((n * 4 + pos, d, semi)); pos += d
    return out, len(bars)


class Tune:
    def __init__(self, key):
        self.key = key; self.cn, src = TUNES[key]; self.notes, self.nb = parse(src)

    def mel(self, smap=None, scale=1):
        """smap：{原音级: 新音级}，新音级可以是小数（微分音）。只换音阶，不动走向"""
        out = []
        for b, d, semi in self.notes:
            pc = semi % 12; new = (smap or {}).get(pc, pc)
            out.append((b * scale, d * scale, TONIC + semi + (new - pc)))
        return out


def harmonize(T, mel, cands, final=None, ms=1):
    """每半小节从候选和弦里挑一个：和弦音盖住的旋律时值最多的那个；并列时照候选顺序。最后半小节用 final"""
    out = []
    for i in range(T.nb * 2):
        lo, hi = i * 2 * ms, (i + 1) * 2 * ms; best, bs = cands[0], -1
        for c in cands:
            r, iv = chord(c); tones = {(r + x) % 12 for x in iv}
            sc = sum(min(b + d, hi) - max(b, lo) for b, d, p in mel if b < hi and b + d > lo and int(round(p - TONIC)) % 12 in tones)
            if (r % 12) == int(round(next((p for b, d, p in mel if lo <= b < hi), TONIC) - TONIC)) % 12: sc += 0.3
            if sc > bs + 1e-6: best, bs = c, sc
        out.append(best)
    if final: out[-1] = final; out[-2] = final if T.notes[-1][0] < (T.nb * 2 - 1) * 2 else out[-2]
    return out


def step(p, sc, k):
    """音阶 sc（半音集合）里，p 往上（k>0）或往下最近的一个音"""
    r = range(1, 8) if k > 0 else range(-1, -8, -1)
    return next(int(round(p)) + q for q in r if (int(round(p)) + q - TONIC) % 12 in sc)


def hb(T, ms=1): return [(i, i * 2 * ms) for i in range(T.nb * 2)]
def mel_put(s, tr, mel, vel=104, oct=0, leg=.95):
    for b, d, p in mel: s.add(tr, b, d * leg, p + 12 * oct, vel + (6 if b % 4 == 0 else 0))
def name(T, st): return f'w_{T.key}_{st}'


# ---------------------------------------------------------------- 各站
def china(T, bpm=92):
    """原版：五声，筝主奏，竖琴分解和弦，下半段笛子高八度跟"""
    s = Song(name(T, '00_china'), bpm); mel = T.mel(); ch = harmonize(T, mel, ['C', 'G', 'Am', 'F'], 'C')
    s.track('mel', 107, vol=120); mel_put(s, 'mel', mel)
    s.track('fl', 73, vol=84); [s.add('fl', b, d * .95, p + 12, 72) for b, d, p in mel if b >= T.nb * 2]
    s.track('comp', 46, vol=82); s.track('bass', 32, vol=90)
    for i, b in hb(T):
        r, iv = chord(ch[i]); v = voicing(ch[i], 55, n=3)
        for k, p in enumerate((v + v[::-1])[:4]): s.add('comp', b + k * .5, .9, p, 62 - k * 3)
        s.add('bass', b, 1.9, near(r, 41), 84)
    s.meta = dict(cn='中国', scale=[0, 2, 4, 7, 9], smap={}, chords=ch)
    return s


def scale_only(T, st, cn, smap, bpm=92):
    """实验版：编曲和原版一模一样，只把音阶换掉。用来证明"变的只有这几个音" """
    s = Song(name(T, st + '_a'), bpm); mel = T.mel(smap); ch = harmonize(T, T.mel(), ['C', 'G', 'Am', 'F'], 'C')
    f = lambda pc: int(round(smap.get(pc % 12, pc % 12)))
    s.track('mel', 107, vol=120); mel_put(s, 'mel', mel); s.track('comp', 46, vol=82); s.track('bass', 32, vol=90)
    for i, b in hb(T):
        r, iv = chord(ch[i]); v = sorted({near(f(r + x), 55) for x in iv})
        for k, p in enumerate((v + v[::-1])[:4]): s.add('comp', b + k * .5, .9, p, 62 - k * 3)
        s.add('bass', b, 1.9, near(f(r), 41), 84)
    s.meta = dict(cn=cn + '·只换音阶', smap=smap)
    return s


JP = {2: 1, 4: 5, 9: 8, 11: 0}


def japan(T, bpm=66):
    """都节音阶（挪三个音）：慢、留白，筝的倚音和句尾刮奏，尺八长音，三味线，太鼓"""
    s = Song(name(T, '01_japan'), bpm); mel = T.mel(JP); sc = [0, 1, 5, 7, 8]
    s.track('mel', 107, vol=122); mel_put(s, 'mel', mel, vel=100)
    for b, d, p in mel:
        if d >= 1 and b % 4 == 0: s.add('mel', b - .12, .12, step(p, sc, 1), 80)
    s.track('shaku', 77, vol=96); s.track('koto2', 107, vol=78); s.track('taiko', 116, vol=112); s.track('sham', 106, vol=96); s.track('bell', 112, vol=58)
    for b, d, p in mel:
        if d >= 1 and b % 2 == 1: s.add('sham', b + .5, .3, p - 12, 84)
    for n in range(T.nb):
        b = n * 4; s.add('shaku', b, 3.6, TONIC + (7 if n % 2 == 0 else 0), 58)
        for k, x in enumerate([-12, -5, 0, 1, 5]): s.add('koto2', b + 2 + k * .12, 1.6, TONIC - 12 + x + 12 * (k > 2), 56)
        if n % 2 == 1:
            for k, p in enumerate([TONIC + 24 + x for x in (8, 7, 5, 1, 0)] + [TONIC + 12 + x for x in (8, 7, 5, 1, 0)]): s.add('koto2', b + 3 + k * .09, .5, p, 60 - k * 2)
        s.add('taiko', b, .8, 43, 98 if n % 2 == 0 else 72)
        if n % 4 == 3: s.add('taiko', b + 3, .4, 43, 80); s.add('taiko', b + 3.5, .4, 43, 90)
        if n % 4 == 0: s.add('bell', b, 2, TONIC + 31, 50)
    s.meta = dict(cn='日本', scale=sc, smap=JP)
    return s


SLENDRO = {2: 2.4, 4: 4.8, 7: 7.2, 9: 9.6}


PELOG = {2: 1, 4: 3, 9: 8, 11: 10}


def indonesia(T, bpm=116, progs=(10, 9, 14), smap=None, dense=True, cym=True):
    """甘美兰：默认用八度五等分的斯连德罗音阶（五个音都不在钢琴上），金属排琴齐奏旋律，高音区两个音来回交错，大锣压句"""
    smap = SLENDRO if smap is None else smap
    s = Song(name(T, '02_indonesia'), bpm); mel = T.mel(smap); deg = sorted({smap.get(x, x) for x in (0, 2, 4, 7, 9)})
    up = lambda p: TONIC + min((x + o for o in (-12, 0, 12, 24) for x in deg if x + o > p - TONIC + 0.1))
    s.track('mel', progs[0], vol=118); s.track('kotek', progs[1], vol=92); s.track('gong', progs[2], vol=120); s.track('kempul', 12, vol=84); s.track('kendang', 0, drum=True, vol=92)
    q = .125 if dense else .25
    for b, d, p in mel:
        s.add('mel', b, d * .98, p, 104); s.add('mel', b, d * .98, p + 12, 84); s.add('mel', b, min(d, 1) * .9, p - 12, 70)
        for k in range(int(round(d / q))): s.add('kotek', b + k * q, q * .8, (p if k % 2 == 0 else up(p)) + 24, 70 if k % 2 == 0 else 58)   # 交错音型
    for n in range(T.nb):
        b = n * 4
        if n % 2 == 0: s.add('gong', b, 7.5, TONIC - 24, 112)
        for k in (1, 3): s.add('kempul', b + k, .9, TONIC - 12 + (deg[3] if k == 1 else 0), 70)
        for off, note, v in ((0, 64, 90), (.75, 62, 62), (1.5, 63, 74), (2, 64, 84), (2.5, 62, 60), (3, 63, 76), (3.5, 62, 66), (3.75, 62, 56)): s.add('kendang', b + off, .2, note, v)
        if cym: [s.add('kendang', b + k * .25, .08, 42, 58 if k % 2 else 72) for k in range(16)]                        # 小镲一直碎碎地响
    s.meta = dict(cn='印尼', scale=deg, smap=smap)
    return s


IN = {9: 8}


def india(T, bpm=90):
    """6 降半音（北印度的 Nat Bhairav 拉格用到的音）：西塔琴每个音都从下面滑上来，长音加回音，坦布拉琴持续低音，塔布拉鼓十六拍循环"""
    s = Song(name(T, '03_india'), bpm); mel = T.mel(IN); sc = [0, 2, 4, 5, 7, 8, 11]
    s.track('mel', 104, vol=124); s.tr['mel']['glide'] = 1.6; s.tr['mel']['glide_min'] = 0.45
    for b, d, p in mel:
        if d >= 1.5: s.add('mel', b, .5, p, 110); s.add('mel', b + .5, .25, step(p, sc, 1), 86); s.add('mel', b + .75, d - .8, p, 100)   # 回音
        else: s.add('mel', b, d * .97, p, 108)
    s.track('chik', 104, vol=70); s.track('tan', 104, vol=84); s.track('bans', 73, vol=66); s.track('tabla', 0, drum=True, vol=104); s.track('harp', 46, vol=70)
    on = {round(b * 2) for b, d, p in mel}
    for e in range(T.nb * 8):                                    # 旋律空着的半拍，高音共鸣弦拨一下主音
        if e not in on: s.add('chik', e * .5, .3, TONIC + 24, 54); s.add('chik', e * .5 + .25, .2, TONIC + 24, 42)
    for n in range(T.nb):
        b = n * 4
        for k, x in enumerate((-5, 0, 0, -12)): s.add('tan', b + k, 1.9, TONIC - 12 + x, 66)            # 坦布拉：5 1 1 低1
        bols = [('dha', 0), ('dhin', .5), ('dhin', 1), ('dha', 1.5), ('dha', 2), ('tin', 2.5), ('tin', 3), ('ta', 3.5)] if n % 2 else [('dha', 0), ('dhin', .5), ('dhin', 1), ('dha', 1.5), ('dha', 2), ('dhin', 2.5), ('dhin', 3), ('dha', 3.5)]
        for bol, off in bols:
            if bol in ('dha', 'dhin'): s.add('tabla', b + off, .3, 64, 96 if bol == 'dha' else 78)
            s.add('tabla', b + off, .15, 60 if bol != 'tin' else 61, 90 if bol in ('dha', 'ta') else 70)
        s.add('tabla', b + 3.75, .1, 60, 60)
    for k, x in enumerate([0, 2, 4, 5, 7, 8, 11, 12, 14, 16]): s.add('harp', -0.0 + k * .07, 1.2, TONIC + x, 52 + k * 2)       # 开头一把竖琴式的扫弦
    ph = [m for m in mel if m[1] >= 2]
    for b, d, p in ph: [s.add('bans', b + 1 + k * .25, .24, q + 12, 60) for k, q in enumerate((p, step(p, sc, 1), p, step(p, sc, -1)))]
    s.meta = dict(cn='印度', scale=[0, 2, 4, 7, 8], smap=IN)
    return s


ME = {2: 1, 9: 8}


def mideast(T, bpm=96):
    """希贾兹（2、6 降半音，♭2 到 3 是增二度）：小提琴齐奏带颤音，乌德琴轮指，持续低音，马克苏姆手鼓节奏"""
    s = Song(name(T, '04_mideast'), bpm); mel = T.mel(ME); sc = [0, 1, 4, 5, 7, 8, 10]
    s.track('mel', 40, vol=114); s.track('oud', 24, vol=112); s.track('str', 48, vol=92); s.track('ney', 75, vol=70); s.track('drone', 42, vol=74); s.track('dr', 0, drum=True, vol=112)
    for b, d, p in mel:
        if d >= 1.5: trill(s, 'mel', b, d, p, step(p, sc, 1), rate=6, vel=96)
        elif d >= 1: s.add('mel', b, .25, p, 100); s.add('mel', b + .25, .25, step(p, sc, 1), 84); s.add('mel', b + .5, d - .55, p, 96)
        else: s.add('mel', b, d * .95, p, 98)
        s.add('str', b, d * .95, p + 12, 74); s.add('str', b, d * .95, p, 70)
        if int(round(p - TONIC)) % 12 == 4 and d >= 1: s.add('mel', b - .13, .13, p - 3, 86); s.add('oud', b - .13, .13, p - 15, 80)      # 3 前面先蹭一下 ♭2：把增二度亮出来
        n = max(1, int(d * 4))
        for k in range(n): s.add('oud', b + k * d / n, d / n * .9, p - 12, 88 - (k % 2) * 14)
    for n in range(T.nb):
        b = n * 4; s.add('drone', b, 3.9, TONIC - 24, 70); s.add('drone', b, 3.9, TONIC - 17, 58)
        if n % 2 == 1: [s.add('ney', b + 2.5 + k * .25, .24, TONIC + 12 + x, 62) for k, x in enumerate((8, 7, 5, 4, 1, 0))]
        for off, note, v in ((0, 61, 112), (.5, 60, 92), (1.5, 60, 88), (2, 61, 104), (3, 60, 94), (3.5, 54, 70), (1, 54, 52), (2.5, 54, 52), (.25, 54, 40), (3.25, 60, 60)): s.add('dr', b + off, .2, note, v)
    s.meta = dict(cn='中东', scale=[0, 1, 4, 7, 8], smap=ME)
    return s


MIN = {4: 3, 9: 8, 11: 10}


def russia(T, bpm=84, bpm_end=184):
    """小调（3、6 降半音）：手风琴主奏，三角琴轮指，大号"嘣"和弦"嚓"，前四小节一路加速"""
    s = Song(name(T, '05_russia'), bpm, bpm_end=bpm_end); s.acc = 16; mel = T.mel(MIN); ch = harmonize(T, mel, ['Cm', 'G7', 'Fm', 'Ab'], 'Cm')
    s.track('mel', 21, vol=127); mel_put(s, 'mel', mel, vel=110); mel_put(s, 'mel', mel, vel=88, oct=1)
    s.track('bal', 25, vol=92); s.track('acc', 21, vol=54); s.track('tuba', 58, vol=94); s.track('dr', 0, drum=True, vol=80)
    for b, d, p in mel:
        n = max(1, int(d * 4))
        for k in range(n): s.add('bal', b + k * d / n, d / n * .8, p + 12, 70 - (k % 2) * 12)
    for i, b in hb(T):
        r, iv = chord(ch[i]); v = voicing(ch[i], 58, n=3)
        s.add('tuba', b, .8, near(r, 38), 96); s.add('tuba', b + 1, .8, near((r + 7) % 12, 38), 88)
        for off in (.5, 1.5): [s.add('acc', b + off, .35, p, 70) for p in v]; s.add('dr', b + off, .1, 54, 70)
        s.add('dr', b, .2, 36, 80); s.add('dr', b + 1, .2, 36, 72)
    s.meta = dict(cn='俄罗斯', scale=[0, 2, 3, 7, 8], smap=MIN, chords=ch)
    return s


PHR = {2: 1, 4: 3, 9: 8, 11: 10}


def spain(T, bpm=120, tp=True, fast=True):
    """弗里几亚（2、3、6 降半音）+ 安达卢西亚终止（四个和弦一级一级往下走）：吉他扫弦的 3+3+2 重音，击掌，响板，小号，句尾一串下行快速音阶"""
    s = Song(name(T, '06_spain'), bpm); mel = T.mel(PHR); sc = [0, 1, 3, 5, 7, 8, 10]
    ch = harmonize(T, mel, ['Fm', 'Eb', 'Db', 'C'], 'C')
    for i in range(1, len(ch), 4):                                # 每两小节的最后落回主和弦，前一个放降二级：听感上就是"西班牙"
        if i + 2 < len(ch): ch[i + 1], ch[i + 2] = 'Db', 'C'
    s.track('mel', 24, vol=124); s.track('tp', 56, vol=100); s.track('gtr', 24, vol=104); s.track('bass', 32, vol=100); s.track('pal', 0, drum=True, vol=104)
    for b, d, p in mel:
        if d >= (1 if fast else 1.5): [s.add('mel', b + k * .25, .22, p + 12, 100 - (k % 2) * 16) for k in range(int(d * 4))]       # 长音用轮指
        else: s.add('mel', b, d * .95, p + 12, 108)
        if tp and b >= T.nb * 2: s.add('tp', b, d * .93, p, 92)
    for i, b in hb(T):
        r, iv = chord(ch[i]); v = [near(r, 43)] + voicing(ch[i], 57, n=3) + [near(r, 67)]
        pat = [(k / 6, int(k % 3 == 0)) for k in range(12)] if fast else (((0, 1), (.5, 0), (.75, 0), (1, 0), (1.5, 1)) if i % 2 == 0 else ((0, 0), (.5, 0), (1, 1), (1.25, 0), (1.5, 0), (1.75, 0)))
        for off, acc in pat:
            for k, p in enumerate(v if acc else v[1:]): s.add('gtr', b + off + k * .012, .3 if not acc else .45, p, (98 if acc else 66) - k * 2)
        s.add('bass', b, 1.4, near(r, 36), 96)
    for n in range(T.nb):
        b = n * 4
        for off in (0, 1.5, 3): s.add('pal', b + off, .2, 36, 96)                                                    # 卡洪鼓：3+3+2
        for off in ([k * .5 for k in range(8)] if fast else (.5, 1, 2, 2.5, 3.5)): s.add('pal', b + off, .1, 39, 84 if off in (1, 2.5) else 62)   # 击掌
        for k in range(6): s.add('pal', b + 3 + k / 6, .08, 85, 60 + k * 8)                                          # 响板滚奏
        if fast: [s.add('pal', b + k / 3, .08, 85, 50) for k in range(9)]
    for n in range(1, T.nb, 2):
        for k, x in enumerate((12, 10, 8, 7, 5, 3, 1, 0)): s.add('mel', n * 4 + 3 + k * .125, .12, TONIC + x, 84)     # 句尾下行跑动
    s.meta = dict(cn='西班牙', scale=[0, 1, 3, 7, 8], smap=PHR, chords=ch)
    return s


DOR = {4: 3}


def ireland(T, bpm=116):
    """多利亚（3 降半音）+ 六八拍吉格：哨笛和提琴齐奏带回音，手鼓，吉他，风笛持续低音"""
    s = Song(name(T, '07_ireland'), bpm, swing=2 / 3); mel = T.mel(DOR); sc = [0, 2, 3, 5, 7, 9, 10]
    ch = harmonize(T, mel, ['Cm', 'Bb', 'F', 'Gm'], 'Cm')
    jig = []
    for b, d, p in mel:
        if d == 1: jig += [(b, .5, p), (b + .5, .5, step(p, sc, 1) if round(b) % 2 else p)]
        elif d >= 2: jig += [(b, 1, p), (b + 1, .5, step(p, sc, 1)), (b + 1.5, d - 1.5, p)]
        else: jig.append((b, d, p))
    s.track('mel', 72, vol=112); [s.add('mel', b, d * .85, p + 12, 96) for b, d, p in jig]
    s.track('fid', 110, vol=92); [s.add('fid', b, d * .9, p, 78) for b, d, p in jig]
    s.track('pipe', 109, vol=62); s.track('gtr', 25, vol=84); s.track('bass', 32, vol=90); s.track('bod', 0, drum=True, vol=100)
    for n in range(T.nb): s.add('pipe', n * 4, 3.95, TONIC - 12, 60); s.add('pipe', n * 4, 3.95, TONIC - 5, 54)
    for i, b in hb(T):
        r, iv = chord(ch[i]); v = voicing(ch[i], 55, n=3); s.add('bass', b, 1.9, near(r, 38), 86)
        for off, vel in ((0, 72), (.5, 48), (1, 64), (1.5, 48)): [s.add('gtr', b + off, .4, p, vel) for p in v]
    for bt in range(T.nb * 4):
        s.add('bod', bt, .3, 45 if bt % 2 == 0 else 47, 90 if bt % 2 == 0 else 72); s.add('bod', bt + .5, .2, 47, 56); s.add('bod', bt + .25, .1, 47, 42)
    s.meta = dict(cn='爱尔兰', scale=[0, 2, 3, 7, 9], smap=DOR, chords=ch, meter='6/8', jig=True)
    s._mel = jig
    return s


BLU = {2: 3, 4: 5, 9: 10}


def blues(T, bpm=84):
    """小调五声（2、3、6 各挪一格）+ 蓝调音：口琴主奏，5 前面蹭一个 ♭5，三连音摇摆，属七和弦，吉他布基律动"""
    s = Song(name(T, '08_blues'), bpm, swing=2 / 3); mel = T.mel(BLU); ch = harmonize(T, mel, ['C7', 'F7', 'G7'], 'C7')
    s.track('mel', 22, vol=127); mel_put(s, 'mel', mel, vel=114); s.track('mel8', 27, vol=96); mel_put(s, 'mel8', mel, vel=84, oct=-1, leg=.9)
    for b, d, p in mel:
        if int(round(p - TONIC)) % 12 == 7 and d >= 1: s.add('mel', b - .14, .14, p - 1, 88)          # ♭5 → 5
    s.track('gtr', 27, vol=70); s.track('org', 16, vol=46); s.track('bass', 33, vol=100); s.track('dr', 0, drum=True, vol=82)
    for i, b in hb(T):
        r, iv = chord(ch[i]); root = near(r, 43)
        for k in range(4): s.add('gtr', b + k * .5, .45, root, 70); s.add('gtr', b + k * .5, .45, root + (7 if k % 2 == 0 else 9), 66)
        [s.add('org', b, 1.9, p, 46) for p in voicing(ch[i], 60, drop_root=True, n=3)]
        s.add('bass', b, .9, near(r, 36), 94); s.add('bass', b + 1, .9, near((r + (4 if i % 2 == 0 else 7)) % 12, 38), 84)
    for bt in range(T.nb * 4):
        s.add('dr', bt, .3, 42, 62); s.add('dr', bt + .5, .2, 42, 46); s.add('dr', bt, .3, 36 if bt % 2 == 0 else 38, 84 if bt % 2 == 0 else 80)
    s.meta = dict(cn='美国蓝调', scale=[0, 3, 5, 7, 10], smap=BLU, chords=ch)
    return s


STATIONS = dict(china=china, japan=japan, indonesia=indonesia, india=india, mideast=mideast, russia=russia, spain=spain, ireland=ireland, blues=blues)
ORDER = ['japan', 'indonesia', 'india', 'mideast', 'russia', 'spain', 'ireland', 'blues']
SMAPS = dict(japan=JP, indonesia=SLENDRO, india=IN, mideast=ME, russia=MIN, spain=PHR, ireland=DOR, blues=BLU)


def ab(T, st, bars=2, bpm=92):
    """来回切：原版一小节、换了音阶的同一小节，交替着放，听得最清楚"""
    a, b2 = china(T, bpm), scale_only(T, STATIONS[st](T).name.split('_', 2)[2], '', SMAPS[st], bpm)
    s = Song(name(T, STATIONS[st](T).name.split('_', 2)[2] + '_ab'), bpm)
    for k in range(bars * 2):
        src = a if k % 2 == 0 else b2; n = k // 2
        for nm, t in src.tr.items():
            s.track(nm, t['prog'], t['drum'], t['vol'])
            for b, d, p, v in t['notes']:
                if n * 4 <= b < n * 4 + 4: s.tr[nm]['notes'].append((b - n * 4 + k * 4, min(d, n * 4 + 4 - b), p, v))
    s._beats = bars * 2 * 4; s.meta = dict(cn='来回切', smap=SMAPS[st], ab=True)
    return s


def export(s):
    """沿用 arr.export，只是旋律音符可以由站自己指定（吉格把长音拆开了，画面要按拆开后的画）"""
    arr.export(s)


def relay(T, plan, bpm=92, tag='relay'):
    """整首接龙：plan = [(站, 从第几小节, 到第几小节)]，各站按同一个速度渲染，在小节线上硬切"""
    bar = 4 * 60 / bpm; out = []; notes = []; tmp = HERE / 'skel/_relay_tmp'; tmp.mkdir(parents=True, exist_ok=True); segs = []
    for st, b0, b1 in plan:
        s = STATIONS[st](T); s.bpm, s.bpm_end = bpm, None
        wav = tmp / f'{s.name}.wav'; mid = tmp / f'{s.name}.mid'; s.midi().write(str(mid)); arr.fluid(mid, wav)
        a, _ = sf.read(wav, always_2d=True); i0, i1 = int(b0 * bar * SR), int(b1 * bar * SR)
        last = (st, b0, b1) == tuple(plan[-1]); off = int(plan[0][1] * bar * SR)
        seg = a[i0:i1 + (int(1.5 * SR) if last else 0)].copy(); seg = seg / (np.sqrt(np.mean(seg ** 2)) + 1e-9) * 0.15
        f = 300; seg[:f] *= np.linspace(0, 1, f)[:, None]; seg[-f:] *= np.linspace(1, 0, f)[:, None]
        if last: g = int(1.2 * SR); seg[-g:] *= np.linspace(1, 0, g)[:, None]
        out.append(seg); segs.append(dict(st=s.name, cn=s.meta['cn'], t0=round(b0 * bar, 3), t1=round(b1 * bar, 3)))
        seen = set()
        for b, d, p, v in sorted(s.tr['mel']['notes'], key=lambda n: (n[0], n[2])):
            if round(b, 3) in seen or not (b0 * 4 <= b < b1 * 4) or d < 0.2: continue
            seen.add(round(b, 3)); notes.append(dict(t=round(s.sec(b), 4), d=round(min(s.sec(b + d), b1 * bar) - s.sec(b), 4), p=p, beat=b, style=s.name))
    a = np.concatenate(out); a = a / (np.abs(a).max() + 1e-9) * 0.89; nm = name(T, tag)
    sf.write(HERE / f'skel/{nm}.wav', a.astype(np.float32), SR)
    nbar = plan[-1][2] - plan[0][1]
    json.dump(dict(name=nm, cn='接龙', bpm=bpm, dur=round(nbar * bar, 3), bars=[round(i * bar, 4) for i in range(nbar + 1)], notes=notes, segs=segs, mel_scale=1),
              open(HERE / f'score/{nm}.json', 'w'), ensure_ascii=False, indent=1)
    print(f'{nm:22s} 接龙 {nbar} 小节 共 {nbar * bar:.1f}s 旋律音 {len(notes)}')


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == '--jianpu':                 # world.py --jianpu <英文名> <中文名> "<简谱>" [站名…]
        TUNES[args[1]] = (args[2], args[3]); args = [args[1]] + args[4:]
    key = args[0] if args and args[0] in TUNES else 'molihua'; T = Tune(key); arr.NB = T.nb
    want = [a for a in args if a not in TUNES] or ['china'] + ORDER + ['japan_a', 'japan_ab', 'relay']
    for w in want:
        arr.NB = T.nb
        if w == 'mini': relay(T, [('japan', 0, 2), ('mideast', 2, 4), ('spain', 4, 6), ('blues', 6, 8)], bpm=126, tag='mini')
        elif w == 'relay':
            n = T.nb; cut = [0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 14] if n == 14 else [round(i * n / 10) for i in range(11)]
            relay(T, [(st, cut[i], cut[i + 1]) for i, st in enumerate(['china'] + ORDER + ['china']) if cut[i + 1] > cut[i]])
        elif w.endswith('_ab'): export(ab(T, w[:-3]))
        elif w.endswith('_a'): export(scale_only(T, STATIONS[w[:-2]](T).name.split('_', 2)[2], STATIONS[w[:-2]](T).meta['cn'], SMAPS[w[:-2]]))
        else: export(STATIONS[w](T))
