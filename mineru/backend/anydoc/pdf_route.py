# Copyright (c) Opendatalab. All rights reserved.
"""PDF 走 anydoc 还是 MinerU 的判定。

anydoc 只抽文本层，扫描件和图文混合 PDF 必须交给 MinerU 的 OCR/VLM 流程；
纯文本 PDF 走 anydoc，解析速度是数量级差异。
"""
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from loguru import logger

from mineru.utils.pdf_classify import MAX_SAMPLE_PAGES, classify, get_sample_page_indices
from mineru.utils.pdf_page_id import get_end_page_id
from mineru.utils.pdfium_guard import (
    close_pdfium_child,
    close_pdfium_document,
    open_pdfium_document,
    pdfium_guard,
)


def _page_has_image_objects(page) -> bool:
    for page_object in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=3):
        close_pdfium_child(page_object)
        return True
    return False


def _covers_full_document_without_images(pdf_bytes: bytes, end_page_id) -> bool:
    """anydoc 不支持按页解析，只有整篇解析且采样页无图片时才可用。"""
    pdf_doc = None
    try:
        with pdfium_guard():
            pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
            page_count = len(pdf_doc)
            if get_end_page_id(end_page_id, page_count) != page_count - 1:
                return False

            for page_index in get_sample_page_indices(page_count, MAX_SAMPLE_PAGES):
                page = None
                try:
                    page = pdf_doc[page_index]
                    if _page_has_image_objects(page):
                        logger.debug("PDF contains images, routing to MinerU instead of anydoc")
                        return False
                finally:
                    close_pdfium_child(page)
    finally:
        close_pdfium_document(pdf_doc)
    return True


def can_parse_pdf_with_anydoc(pdf_bytes: bytes, start_page_id: int = 0, end_page_id=None) -> bool:
    if start_page_id != 0:
        return False
    if classify(pdf_bytes) != "txt":
        return False
    return _covers_full_document_without_images(pdf_bytes, end_page_id)
