# Copyright (c) Opendatalab. All rights reserved.
"""探测 OOXML 包里 anydoc 覆盖不到的内容，命中的文档交回 MinerU 原生解析器。

anydoc 的文档模型没有 chart 概念（chartN.xml + 内嵌数据表整块消失），
Word 目录域也只会被当成普通段落，丢掉 index 结构和 _Toc 跳转锚点。
"""
import zipfile
from io import BytesIO

_CHART_DIRS = ("word/charts/", "xl/charts/", "ppt/charts/")
# Word 目录域的 instrText/fldSimple 都写成 ` TOC \o "1-3" \h ` 这种带开关的形式。
_TOC_FIELD = b"TOC \\"


def ooxml_needs_native_parser(file_bytes: bytes) -> bool:
    with zipfile.ZipFile(BytesIO(file_bytes)) as package:
        names = package.namelist()
        if any(name.startswith(_CHART_DIRS) for name in names):
            return True
        if "word/document.xml" in names:
            return _TOC_FIELD in package.read("word/document.xml")
    return False
