Now verify routing: chart/TOC docs → native, plain docs → anydoc.

加好了。

**新增** `mineru/backend/anydoc/ooxml_probe.py`

```python
_CHART_DIRS = ("word/charts/", "xl/charts/", "ppt/charts/")
_TOC_FIELD = b"TOC \\"   # instrText / fldSimple 的目录域开关
```

只读 zip `namelist()`，命中 chart 目录直接返回；否则 docx 再读一次 `word/document.xml` 找目录域。

**接入** `mineru/cli/common.py:648` `_analyze_office_doc` 判定顺序：

1. OOXML 且探测到 chart / 目录域 → 原生解析器
2. 否则 anydoc
3. anydoc 抛 `ConvertError` 且是 OOXML → 回退原生（非 OOXML 直接抛）

**实测**

| 文件 | 探测耗时 | 走向 | 结果 |
|---|---|---|---|
| docx_01（2 chart + TOC）| 0.07ms | native | 3 页，`chart: 2`、`index: 1`，md 内 35 个 `](#_Toc` 锚点 |
| pptx_01（1 chart）| 0.10ms | native | 6 页，`chart: 1` |
| xlsx_01（2 chart）| 0.08ms | native | 3 页（按 sheet 分页），`chart: 2` |
| plain.docx（无 chart/TOC）| 0.06ms | anydoc | 标题/正文/表格齐全 |
| t.rtf / demo3.pdf | — | anydoc | 不受影响 |

探测开销 0.06–0.1ms，相对 native 解析的 20–210ms 可忽略。

**当前完整分流**

- office 后缀 → 有 chart/TOC 的 OOXML 走 native，其余走 anydoc（失败回退 native）
- doc/ppt/xls/odt/ods/odp/rtf/epub/csv → anydoc（无原生实现，失败即报错）
- PDF 纯文本层 + 整篇 → anydoc；扫描件/含图/指定页范围 → 原 backend
- 图片一律落盘后走 `analyze_office_images` / `aio_analyze_office_images` 的 VLM 入口出 caption

剩余已知差异只有：走 anydoc 的那部分 office 文档仍是单页（`page_idx` 恒 0）。