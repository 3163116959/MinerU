office 文档全部交给 anydoc，原生 OOXML 解析器已整体移除。

**分流**

- office 后缀（docx/pptx/xlsx/doc/ppt/xls/odt/ods/odp/rtf/epub/csv）→ anydoc，解析失败直接报错
- PDF 纯文本层 + 整篇 → anydoc；扫描件 / 含图 / 指定页范围 → MinerU 原 backend（`pdf_route.can_parse_pdf_with_anydoc`）
- 图片解码落盘到 `<output>/<name>/office/images/`，再由 `analyze_office_images` / `aio_analyze_office_images` 出 VLM caption

**链路** `mineru/cli/common.py`

1. `_parse_office_docs` → `anydoc_analyze(file_bytes, suffix, image_writer=...)`
2. 非 PDF：`anydoc.to_document` → `document_to_page_blocks`；PDF：`to_markdown_bytes` → `markdown_to_page_blocks`
3. 统一汇入 office 中间层 `result_to_middle_json`，图片 data-URI 在此解码落盘
4. `_write_parsed_outputs` → `_process_output` 出 md / content_list / middle_json

**实测**（`demo/office_docs`）

| 文件 | 耗时 | 块统计 |
|---|---|---|
| docx_01 | 0.01s | title 37、text 64、table 10、image 5、list 5、interline_equation 21 |
| pptx_01 | 0.00s | title 4、text 12、table 2、list 9、image 1 |
| xlsx_01 | 0.00s | title 3、table 3 |

图片均落盘且 md 内以 `![](images/<sha256>.<ext>)` 引用。

**已知取舍**

- chart 全丢：anydoc 文档模型无 chart 概念，`chartN.xml` 及内嵌数据表不进结果
- Word 目录域降级为普通段落，无 `index` 块、无 `_Toc` 锚点
- anydoc 无分页概念，office 文档恒为单页（`page_idx` 为 0）

**随之移除**

`mineru/model/{docx,pptx,xlsx}/`、`mineru/model/office_stream.py`、
`mineru/backend/office/{docx,pptx,xlsx}_analyze.py`、`mineru/backend/anydoc/ooxml_probe.py`、
`mineru/backend/utils/office_chart.py`，以及仅服务于它们的依赖
python-docx / pypptx-with-oxml / mammoth / pylatexenc / lxml / openpyxl。
