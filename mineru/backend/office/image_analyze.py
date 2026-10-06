# Copyright (c) Opendatalab. All rights reserved.
"""Office 文档没有页面位图，走不了版面 VLM，图片内容对检索完全不可见。

这里对已提取落盘的图片单独调 VLM 单图分析，把结果写进 image span 的 content，
与 hybrid 的 VLM 图片描述同构，md 渲染为同样的 <details> 块。
chart 不处理：chart 的数据已经以 HTML 表格形式落在 span 里，本身可检索。
"""
import base64
from io import BytesIO
from pathlib import Path
from typing import NamedTuple

from PIL import Image
from loguru import logger

from mineru.backend.utils.office_image import get_standard_vector_placeholder_data_uri
from mineru.backend.vlm.vlm_analyze import (
    ModelSingleton,
    aio_predictor_execution_guard,
    predictor_execution_guard,
)
from mineru.utils.enum_class import BlockType, ContentType


class _Target(NamedTuple):
    span: dict
    image: Image.Image


def _placeholder_bytes() -> bytes:
    return base64.b64decode(get_standard_vector_placeholder_data_uri().split(",", 1)[1])


def _first_image_span(body: dict) -> dict | None:
    for line in body.get("lines") or []:
        for span in line.get("spans") or []:
            if span.get("type") == ContentType.IMAGE and span.get("image_path"):
                return span
    return None


def _iter_caption_free_image_bodies(middle_json: dict):
    """只挑没有图注的 image 块：原文自带图注时不覆盖。"""
    for page in middle_json.get("pdf_info") or []:
        for block in page.get("para_blocks") or []:
            if block.get("type") != BlockType.IMAGE:
                continue
            sub_blocks = block.get("blocks") or []
            if any(sub.get("type") == BlockType.IMAGE_CAPTION for sub in sub_blocks):
                continue
            for body in sub_blocks:
                if body.get("type") == BlockType.IMAGE_BODY:
                    yield body
                    break


def _collect_targets(middle_json: dict, image_dir: Path) -> list[_Target]:
    placeholder = _placeholder_bytes()
    targets: list[_Target] = []
    for body in _iter_caption_free_image_bodies(middle_json):
        span = _first_image_span(body)
        if span is None:
            continue
        image_bytes = (image_dir / span["image_path"]).read_bytes()
        if image_bytes == placeholder:
            # 占位图没有信息，送去分析纯属浪费 token
            continue
        with Image.open(BytesIO(image_bytes)) as image:
            targets.append(_Target(span, image.convert("RGB")))
    return targets


def _build_predictor(vlm_backend: str, server_url: str | None, client_kwargs: dict):
    # server_url 只对 *-client 后端有意义，本地引擎带上会污染 predictor 缓存键
    if not vlm_backend.endswith("client"):
        server_url = None
    return ModelSingleton().get_model(vlm_backend, None, server_url, **client_kwargs)


def _fill_span_contents(targets: list[_Target], contents) -> int:
    captioned = 0
    for target, content in zip(targets, contents):
        text = (content or "").strip()
        if not text:
            continue
        target.span["content"] = text
        captioned += 1
    return captioned


def _log_result(captioned: int, total: int) -> None:
    logger.info(f"office image analysis: {captioned}/{total} images captioned")


def analyze_office_images(
    middle_json: dict,
    image_dir: str | Path,
    *,
    vlm_backend: str,
    server_url: str | None = None,
    **client_kwargs,
) -> int:
    targets = _collect_targets(middle_json, Path(image_dir))
    if not targets:
        return 0

    predictor = _build_predictor(vlm_backend, server_url, client_kwargs)
    with predictor_execution_guard(predictor):
        contents = predictor.batch_content_extract(
            [target.image for target in targets], types="image"
        )

    captioned = _fill_span_contents(targets, contents)
    _log_result(captioned, len(targets))
    return captioned


async def aio_analyze_office_images(
    middle_json: dict,
    image_dir: str | Path,
    *,
    vlm_backend: str,
    server_url: str | None = None,
    **client_kwargs,
) -> int:
    targets = _collect_targets(middle_json, Path(image_dir))
    if not targets:
        return 0

    predictor = _build_predictor(vlm_backend, server_url, client_kwargs)
    async with aio_predictor_execution_guard(predictor):
        contents = await predictor.aio_batch_content_extract(
            [target.image for target in targets], types="image"
        )

    captioned = _fill_span_contents(targets, contents)
    _log_result(captioned, len(targets))
    return captioned
