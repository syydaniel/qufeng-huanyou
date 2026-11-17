# 曲风环游 Qufeng-Huanyou

**一首曲子进去，八个国家的版本出来。**

给它一首简单旋律的简谱，它做出日本、印尼、印度、中东、俄罗斯、西班牙、爱尔兰、美国蓝调八个版本，外加整首接龙和自检指标。

做法只有一条规矩：**旋律的走向不动，只挪几个音**。先换成那个国家的音阶，再配上那里的节奏、和声和乐器。比如《茉莉花》的 1 2 3 5 6，挪成 1 ♭2 4 5 ♭6 就到了日本，3 和 6 各降半音就到了俄罗斯。

曲风环游是一个给 AI 助手（Claude Code、Codex 等）用的 skill，流程是：

1. **写简谱**：1–7 是音级，`'` 高八度，`_` 半拍，`|` 分小节，写完对着公开曲谱逐音核对。
2. **渲染各国版本**：纯音符版，旋律 100% 准确；每站还会输出 MIDI，以及每个旋律音的时间和音高（给画面用）。
3. **上色**（可选）：用 ACE-Step 1.5 照着纯音符版重奏，让音色更真。
4. **自检**：量旋律命中率、对照组、时间偏移和响度，再用 CLAP 给风格排个名次。
5. **交给人听**：像不像那个国家，指标证明不了，最后由你的耳朵定。

## 安装

把这个仓库的地址发给你的 AI 助手，跟它说「帮我安装这个 skill」。或者手动：

```bash
git clone https://github.com/syydaniel/qufeng-huanyou ~/.claude/skills/qufeng-huanyou
brew install fluid-synth
python3 -m venv ~/.venvs/huanyou && ~/.venvs/huanyou/bin/pip install numpy soundfile pretty_midi librosa pyloudnorm
```

然后跟 AI 说：「用曲风环游，把《小星星》送去八个国家。」

## 需要什么

- `fluidsynth`，和一个装了 `numpy soundfile pretty_midi` 的 Python（自检还要 `librosa pyloudnorm`）。
- 音色库 MuseScore_General.sf3 已经放在 `sf/` 里（MIT 授权）。
- 可选：[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)，用来上色。装在别处的话，设环境变量 `ACESTEP_ROOT`。
- 只在苹果芯片的 Mac 上跑过。纯音符版的部分应该哪里都能跑，上色那一步要看 ACE-Step 支不支持你的机器。

## 目录

```
SKILL.md        给 AI 看的总说明：简谱写法、每站动了什么、怎么加一个国家
scripts/        world.py 渲染各国版本，arr.py 编曲，cover.py 上色，check.py 自检，style.py 风格打分
sf/             音色库和它的许可证
```

## 已知的不足

- 只处理四四拍、单旋律。
- 每个国家只取了一种最有代表性的音阶，是简化过的。
- 音色库里没有古筝、甘美兰、乌德琴的专门音色，用的是相近的，这几站更依赖上色。
- 所有声音都只过了程序指标，每一站像不像，欢迎指正。

## 同一系列的 skill

- [卡字 kazi](https://github.com/syydaniel/kazi)：让讲解动画的画面卡在旁白的关键词上。
- [开口 kaikou](https://github.com/syydaniel/kaikou)：让克隆的声音，一开口就说人话。
- [套娃 taowa](https://github.com/syydaniel/taowa)：把「讲一个 skill 的视频」从选题做到发布包。
- [真实感开关 zhenshi-kaiguan](https://github.com/syydaniel/zhenshi-kaiguan)：用纯代码做出看着像拍出来的 3D 画面。

## 许可

MIT。音色库 MuseScore_General 也是 MIT，见 `sf/MuseScore_General_License.md`。要发布的作品，请用公有领域的曲子，或者你有权使用的曲子。
