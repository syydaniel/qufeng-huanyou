"""把骨架音频喂给 ACE-Step 1.5 的翻唱模式"上色"。
用法： $ACESTEP_ROOT/.venv/bin/python scripts/cover.py <模型> <站名[:强度[:加噪[:种子[:模式]]]]> …
  模型： turbo | xl-turbo | xl-sft；后面加 -pt 表示不用 MLX、走 PyTorch（MLX 那条路不支持"加噪"参数）
  强度： 前多少比例的步数参考源音频；加噪：从源音频加噪后起步，越大越贴近源（只有 -pt 生效）
  模式： c = 翻唱（源音频先压成 5Hz 语义码，旋律细节会丢）  n = 不量化的翻唱（直接用源音频的潜变量，更贴源）
  例：  cover.py turbo 01_japan:0.6:0.0:11:n 03_jazz:0.6:0.15:11
产物： <HUANYOU_OUT>/cover/<站>__<模型><模式>_s<强度>_n<加噪>_<种子>.wav
"""
import json, os, re, shutil, sys, time
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(k, None)
ROOT = os.environ.get("ACESTEP_ROOT") or os.path.expanduser("~/ACE-Step-1.5"); sys.path.insert(0, ROOT)
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
HERE = os.environ.get('HUANYOU_OUT') or os.path.join(os.getcwd(), 'huanyou_out')
MODELS = {'turbo': ('acestep-v15-turbo', 8), 'xl-turbo': ('acestep-v15-xl-turbo', 8), 'xl-sft': ('acestep-v15-xl-sft', 50)}

# 每站：(调, 风格描述)。速度和时长从 score/<站>.json 读
CAP = {
 '00_china': ('C major', 'Traditional Chinese folk instrumental. A solo guzheng plucks the melody clearly with gentle glissando, a bamboo dizi flute doubles softly, light pipa accompaniment. Calm, elegant, Jiangnan water town, warm intimate recording. No vocals, no drums.'),
 '01_japan': ('C minor', 'Traditional Japanese instrumental. A solo koto plays the melody with bends and grace notes, a breathy shakuhachi holds long notes, shamisen plucks, a deep taiko drum hits sparsely, lots of silence and space. Serene Kyoto temple garden. No vocals.'),
 '02_indonesia': ('C major', 'Balinese and Javanese gamelan orchestra instrumental. Bronze metallophones and gangsa play the melody with shimmering interlocking kotekan patterns, bonang gong chimes, a huge deep gong marks each phrase, kendang hand drums. Hypnotic, bright metallic, temple ceremony. No vocals.'),
 '03_india': ('C major', 'North Indian classical raga instrumental. A sitar plays the melody with sliding meend bends and sympathetic strings shimmering, a tanpura drone hums constantly, tabla plays a teental groove, a bansuri flute answers. Meditative then lively, Hindustani music. No vocals.'),
 '04_mideast': ('C minor', 'Middle Eastern Arabic instrumental in Hijaz maqam. A violin section plays the melody in unison with trills and ornaments, an oud doubles with fast tremolo picking, darbuka plays a maqsum rhythm with riq tambourine, a low drone. Desert caravan, bazaar. No vocals.'),
 '05_russia': ('C minor', 'Russian folk dance instrumental that keeps accelerating. A bright accordion plays the melody, balalaika tremolo doubles it, tuba oom-pah bass, tambourine. Starts slow and heavy and speeds up to a frantic finish, like Kalinka. No vocals.'),
 '06_spain': ('C minor', 'Spanish flamenco instrumental. A nylon flamenco guitar plays the melody with fast picado runs and tremolo, fierce rasgueado strumming, hand claps palmas, castanets, cajon, a bold paso doble trumpet joins. Andalusian cadence, fiery, bullring. No vocals.'),
 '07_ireland': ('C minor', 'Lively Irish jig in 6/8 in Dorian mode. A tin whistle plays the melody with rolls and cuts, a fiddle doubles, uilleann pipes drone, acoustic guitar strums, bodhran frame drum drives the dance. Dublin pub session, joyful Celtic folk. No vocals.'),
 '08_blues': ('C minor', 'Slow Chicago shuffle blues instrumental. A wailing blues harmonica plays the melody with bends, gritty electric guitar boogie riff, Hammond organ pads, electric bass, laid-back shuffle drums. Smoky bar, 12/8 feel. No vocals.'),
}
capof = lambda name: CAP[re.search(r'(\d\d_[a-z]+)', name).group(1)]

model = sys.argv[1]; pt = model.endswith('-pt'); cfg, steps = MODELS[model.replace('-pt', '')]
dit = AceStepHandler(); print(dit.initialize_service(project_root=ROOT, config_path=cfg, device="auto", offload_to_cpu=False, use_mlx_dit=not pt), flush=True)
lm = LLMHandler()          # 翻唱不用语言模型，传一个没初始化的进去
fields = GenerationParams.__dataclass_fields__
os.makedirs(os.path.join(HERE, 'cover'), exist_ok=True)
for job in ' '.join(sys.argv[2:]).split():
    parts = job.split(':'); name = parts[0]
    st = float(parts[1]) if len(parts) > 1 else 0.6; nz = float(parts[2]) if len(parts) > 2 else 0.0; seed = int(parts[3]) if len(parts) > 3 else 11; mode = parts[4] if len(parts) > 4 else 'c'; src = parts[5] if len(parts) > 5 else ''   # src='mel'：只喂旋律单轨
    sc = json.load(open(os.path.join(HERE, 'score', name + '.json'))); key, cap = capof(name)
    kw = dict(task_type='cover' if mode == 'c' else 'cover-nofsq', src_audio=os.path.join(HERE, 'skel', name + ('_mel' if src == 'mel' else '') + '.wav'), caption=cap, lyrics='[Instrumental]', instrumental=True,
              bpm=int(round(sc['bpm'] if not sc.get('bpm_end') else (sc['bpm'] + sc['bpm_end']) / 2)), keyscale=key, timesignature='6' if sc.get('meter') == '6/8' else '4',
              vocal_language='unknown', duration=max(10.0, round(sc['dur'] + 1.0, 1)), inference_steps=steps, guidance_scale=1.0 if 'turbo' in model else 7.0,
              seed=seed, thinking=False, audio_cover_strength=st)
    if 'cover_noise_strength' in fields: kw['cover_noise_strength'] = nz
    for f in ('use_cot_metas', 'use_cot_caption', 'use_cot_language'):
        if f in fields: kw[f] = False
    t0 = time.time()
    r = generate_music(dit, lm, params=GenerationParams(**kw), config=GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format='wav'),
                       save_dir=os.path.join(HERE, 'cover', '_raw'))
    out = os.path.join(HERE, 'cover', f'{name}__{model}{mode}{src}_s{st}_n{nz}_{seed}.wav')
    if r.success and r.audios: shutil.copyfile(r.audios[0]['path'], out)
    print(name, model, mode, st, nz, seed, r.success, round(time.time() - t0), 's', os.path.basename(out) if r.success else r.status_message, flush=True)
