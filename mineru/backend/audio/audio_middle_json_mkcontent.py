# Copyright (c) Opendatalab. All rights reserved.
"""把音频 middle_json 渲染为 md（按说话人合并的可读文稿）与 content_list（逐段结构化）。"""


def _format_ts(seconds) -> str:
    total = int(seconds or 0)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def make_markdown(middle_json: dict) -> str:
    """相邻同一说话人的段落合并为一段，便于阅读。"""
    turns: list[dict] = []
    for seg in middle_json["segments"]:
        if turns and turns[-1]["speaker"] == seg["speaker"]:
            turns[-1]["end"] = seg["end"]
            turns[-1]["texts"].append(seg["text"])
        else:
            turns.append({**seg, "texts": [seg["text"]]})

    blocks = [
        f"**{turn['speaker'] or 'UNKNOWN'}** [{_format_ts(turn['start'])} - {_format_ts(turn['end'])}]\n\n"
        f"{''.join(turn['texts'])}"
        for turn in turns
    ]
    return "\n\n".join(blocks) + "\n"


def make_content_list(middle_json: dict) -> list[dict]:
    """逐段输出，不含词级信息（词级见 middle_json）。"""
    return [
        {
            "type": "speech",
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
        }
        for seg in middle_json["segments"]
    ]
