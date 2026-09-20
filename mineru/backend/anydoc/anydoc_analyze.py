# Copyright (c) Opendatalab. All rights reserved.
"""anydoc 解析入口：只做结构解析，图片内容仍由 MinerU 的 VLM 入口统一处理。"""
import time

import anydoc
from loguru import logger

from mineru.backend.anydoc.document_to_blocks import document_to_page_blocks
from mineru.backend.anydoc.markdown_to_blocks import markdown_to_page_blocks
from mineru.backend.office.model_output_to_middle_json import result_to_middle_json

AnydocConvertError = anydoc.ConvertError


def anydoc_format(file_suffix: str) -> str | None:
    """把 MinerU 的文件后缀映射为 anydoc 格式名，不支持时返回 None。"""
    return anydoc.format_from_extension(f".{file_suffix}")


def anydoc_analyze(file_bytes: bytes, file_suffix: str, image_writer=None):
    """解析单个文档，返回与 office 后端一致的 (middle_json, results)。"""
    file_format = anydoc_format(file_suffix)
    if file_format is None:
        raise ValueError(f"anydoc does not support suffix: {file_suffix}")

    infer_start = time.time()
    if file_format == "pdf":
        # PDF 在 anydoc 里只有 Markdown 形态，没有文档模型。
        page_blocks = markdown_to_page_blocks(
            anydoc.to_markdown_bytes(file_bytes, file_format)
        )
    else:
        page_blocks = document_to_page_blocks(
            anydoc.to_document(file_bytes, file_format)
        )
    logger.debug(
        f"anydoc parse finished: format={file_format}, "
        f"blocks={len(page_blocks)}, cost={round(time.time() - infer_start, 2)}s"
    )

    results = [page_blocks]
    return result_to_middle_json(results, image_writer), results
