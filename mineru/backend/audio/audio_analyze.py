# Copyright (c) Opendatalab. All rights reserved.
"""音频解析入口：Qwen3-ASR 转写 + VAD + 词级对齐 + 说话人分离。"""
import math
import os
import re
import tempfile
import threading
import time

from loguru import logger
from wetext import Normalizer

from mineru.utils.config_reader import get_device, get_local_models_dir
from mineru.utils.enum_class import AudioModelPath
from mineru.version import __version__

LANGUAGE = "zh"
# 中文语速快、无空格分词，30s 窗口对齐时漂移明显；15s 窗口在时间戳精度与上下文间更平衡。
CHUNK_SIZE = 15
VAD_ONSET = 0.5
VAD_OFFSET = 0.363
BATCH_SIZE = 8

# Qwen3-ASR 贴近读音转写（"三三零零零"），ITN 规整为书面数字（"33000"）。
_itn = Normalizer(lang=LANGUAGE, operator="itn")
# Qwen3-ASR 把逐字母念的缩写输出为 "P D C A"，仅合并单个大写字母间的空格。
_SPACED_LETTERS = re.compile(r"(?<=\b[A-Z]) (?=[A-Z]\b)")

_models = None
# 模型常驻且非线程安全；串行推理同时避免与 vllm 争抢显存时并发放大峰值。
_lock = threading.Lock()


def _audio_models_root() -> str:
    """从 mineru.json 读取音频模型根目录，缺失直接报错，运行期不允许联网下载。"""
    models_dir = get_local_models_dir() or {}
    root = models_dir.get("audio")
    if not root or not os.path.isdir(root):
        raise RuntimeError(
            "Audio models not found, run `mineru-models-download -m audio` first."
        )
    return root


def _load_models():
    """首次调用时加载 ASR/对齐/分离三套模型，进程内复用。"""
    global _models
    if _models is not None:
        return _models

    root = _audio_models_root()
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")

    import whisperx
    from whisperx.asr_qwen import load_model as load_qwen_model
    from whisperx.alignment_qwen import load_align_model as load_qwen_align_model
    from whisperx.diarize import DiarizationPipeline

    device = "cuda" if get_device().startswith("cuda") else "cpu"
    qwen_dtype = "float16" if device == "cuda" else "float32"

    asr_model = load_qwen_model(
        os.path.join(root, AudioModelPath.qwen_asr[1]),
        device,
        language=LANGUAGE,
        vad_method="pyannote",
        vad_options={"chunk_size": CHUNK_SIZE, "vad_onset": VAD_ONSET, "vad_offset": VAD_OFFSET},
        local_files_only=True,
        qwen_dtype=qwen_dtype,
    )
    align_model, align_metadata = load_qwen_align_model(
        LANGUAGE,
        device,
        model_name=os.path.join(root, AudioModelPath.qwen_align[1]),
        model_cache_only=True,
        qwen_dtype=qwen_dtype,
    )
    diarize_model = DiarizationPipeline(
        model_name=os.path.join(root, AudioModelPath.diarize[1]), device=device
    )
    _models = (device, asr_model, align_model, align_metadata, diarize_model)
    return _models


def _clean_float(value):
    """对齐失败的词 score 可能是 NaN，json 不接受 NaN → 置 None。"""
    if value is None:
        return None
    value = float(value)
    return round(value, 3) if math.isfinite(value) else None


def _postprocess_text(text: str) -> str:
    return _SPACED_LETTERS.sub("", _itn.normalize(text.strip()))


def _build_segments(aligned: dict) -> list[dict]:
    """把 whisperx 结果整理为稳定的段落结构，词级时间戳与说话人一并保留。

    文本规整只作用于段落 text：对齐器需读音文本，必须在对齐后做；
    规整后字数变化无法逐字映射，words 保留读音原文。
    """
    segments = []
    for seg in aligned["segments"]:
        words = [
            {
                "word": word["word"],
                "start": _clean_float(word.get("start")),
                "end": _clean_float(word.get("end")),
                "score": _clean_float(word.get("score")),
                "speaker": word.get("speaker"),
            }
            for word in seg.get("words", [])
        ]
        segments.append({
            "start": _clean_float(seg["start"]),
            "end": _clean_float(seg["end"]),
            "speaker": seg.get("speaker"),
            "text": _postprocess_text(seg["text"]),
            "words": words,
        })
    return segments


def _run(audio_path: str):
    """VAD 切分 → Qwen 转写 → 对齐 → 说话人分离，返回 (对齐结果, 原始转写段, 说话人轮次, 时长)。"""
    import whisperx

    device, asr_model, align_model, align_metadata, diarize_model = _load_models()
    audio = whisperx.load_audio(audio_path)
    duration = len(audio) / whisperx.audio.SAMPLE_RATE

    transcript = asr_model.transcribe(
        audio, batch_size=BATCH_SIZE, chunk_size=CHUNK_SIZE, language=LANGUAGE
    )
    # VAD 判定全程无语音时没有可对齐/分离的内容，直接返回空结果。
    if not transcript["segments"]:
        return {"segments": []}, [], [], duration

    from whisperx.alignment_qwen import align as align_qwen

    aligned = align_qwen(
        transcript["segments"], align_model, align_metadata, audio, device,
        return_char_alignments=False,
    )
    diarize_df = diarize_model(audio)
    aligned = whisperx.assign_word_speakers(diarize_df, aligned)
    turns = [
        {"start": _clean_float(row.start), "end": _clean_float(row.end), "speaker": row.speaker}
        for row in diarize_df.itertuples()
    ]
    return aligned, transcript["segments"], turns, duration


def audio_analyze(file_bytes: bytes, file_suffix: str):
    """解析单个音频，返回 (middle_json, results)；results 为模型原始输出。"""
    infer_start = time.time()
    # ffmpeg 需从文件读取，后缀仅用于帮助 ffmpeg 探测容器格式。
    with tempfile.NamedTemporaryFile(suffix=f".{file_suffix}") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        with _lock:
            aligned, asr_segments, turns, duration = _run(tmp.name)

    segments = _build_segments(aligned)
    middle_json = {
        "_backend": "audio",
        "_version_name": __version__,
        "language": LANGUAGE,
        "duration": _clean_float(duration),
        "speakers": sorted({turn["speaker"] for turn in turns}),
        "segments": segments,
    }
    results = {
        "asr_segments": [
            {"start": _clean_float(s["start"]), "end": _clean_float(s["end"]), "text": s["text"]}
            for s in asr_segments
        ],
        "diarization": turns,
    }
    logger.debug(
        f"audio parse finished: duration={round(duration, 2)}s, "
        f"segments={len(segments)}, cost={round(time.time() - infer_start, 2)}s"
    )
    return middle_json, results
