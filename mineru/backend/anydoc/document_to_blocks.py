# Copyright (c) Opendatalab. All rights reserved.
"""把 anydoc 文档模型转成 office 中间块，复用 office 中间层、输出层和图片 VLM 分析。"""
import base64
from html import escape

_STYLE_NAMES = (("bold", "bold"), ("italic", "italic"), ("strike", "strikethrough"))
_MARKER_ORDERED = {"decimal", "lower_alpha", "upper_alpha", "lower_roman", "upper_roman"}


def _style_str(style) -> str:
    if style is None:
        return ""
    return ",".join(name for attr, name in _STYLE_NAMES if getattr(style, attr, False))


def _wrap_text(text: str, style) -> str:
    style_str = _style_str(style)
    if not style_str:
        return text
    return f'<text style="{style_str}">{text}</text>'


def _asset_data_uri(source, assets) -> str:
    if source is None or source.kind != "asset" or source.asset_id is None:
        return ""
    asset = assets[source.asset_id]
    return f"data:{asset.media_type};base64,{base64.b64encode(asset.data).decode()}"


def _note_marker(note_id: str, note_numbers: dict[str, int]) -> str:
    number = note_numbers.get(note_id)
    if number is None:
        return ""
    return f'<text style="superscript">[{number}]</text>'


def _render_inline_text(inline, assets, note_numbers: dict[str, int]) -> str:
    """渲染单个非图片 inline 为 office content 文本。"""
    kind = inline.kind
    if kind == "text":
        return _wrap_text(inline.text or "", inline.style)
    if kind == "math":
        return f"<eq>{inline.text or ''}</eq>"
    if kind == "line_break":
        return "\n"
    if kind == "checkbox":
        return "[x] " if inline.checked else "[ ] "
    if kind == "note_ref":
        return _note_marker(inline.note_id or "", note_numbers)
    if kind == "link":
        return _render_link(inline, assets, note_numbers)
    return ""


def _link_label_segment(child, assets, note_numbers: dict[str, int]) -> str:
    """超链接标签内只认 <text> 子片段，未包裹的文本会被 office 解析器丢弃。"""
    rendered = _render_inline_text(child, assets, note_numbers)
    if not rendered or rendered.startswith("<text"):
        return rendered
    return f"<text>{rendered}</text>"


def _render_link(inline, assets, note_numbers: dict[str, int]) -> str:
    children = [child for child in (inline.content or []) if child.kind != "image"]
    target = inline.target
    if target is None or target.kind == "anchor":
        return "".join(
            _render_inline_text(child, assets, note_numbers) for child in children
        )

    label = "".join(
        _link_label_segment(child, assets, note_numbers) for child in children
    )
    if not label:
        return ""
    return f"<hyperlink>{label}<url>{target.value}</url></hyperlink>"


def _image_block(inline, assets) -> dict | None:
    data_uri = _asset_data_uri(inline.source, assets)
    if not data_uri:
        return None
    return {"type": "image", "content": data_uri}


def _inlines_to_blocks(
    inlines,
    assets,
    note_numbers: dict[str, int],
    block_type: str = "text",
    extra: dict | None = None,
) -> list[dict]:
    """把 inline 序列拆成文本块和图片块：office 中间层要求图片独立成块。"""
    blocks: list[dict] = []
    buffer: list[str] = []

    def flush():
        content = "".join(buffer).strip()
        buffer.clear()
        if content:
            blocks.append({"type": block_type, "content": content, **(extra or {})})

    for inline in inlines or []:
        if inline.kind == "image":
            flush()
            image_block = _image_block(inline, assets)
            if image_block:
                blocks.append(image_block)
            continue
        buffer.append(_render_inline_text(inline, assets, note_numbers))

    flush()
    return blocks


def _inlines_to_plain_text(inlines, assets, note_numbers: dict[str, int]) -> str:
    return "".join(
        _render_inline_text(inline, assets, note_numbers)
        for inline in (inlines or [])
        if inline.kind != "image"
    ).strip()


def _cell_block_html(block, assets, note_numbers: dict[str, int]) -> str:
    """渲染单元格内的一个块：文本转义后内联，图片保留 data URI 供中间层落盘。"""
    if block.kind == "table":
        return _table_html(block.table, assets, note_numbers)
    if block.kind in ("code_block", "math"):
        return escape(block.text or "")
    if block.kind == "list":
        return "<br>".join(
            html
            for item in block.list.items
            for item_block in item.blocks
            for html in [_cell_block_html(item_block, assets, note_numbers)]
            if html
        )
    if block.kind == "block_quote":
        return "<br>".join(
            html
            for child in block.blocks or []
            for html in [_cell_block_html(child, assets, note_numbers)]
            if html
        )

    parts: list[str] = []
    for inline in block.content or []:
        if inline.kind == "image":
            data_uri = _asset_data_uri(inline.source, assets)
            if data_uri:
                parts.append(f'<img src="{data_uri}">')
            continue
        parts.append(escape(_render_inline_text(inline, assets, note_numbers)))
    return "".join(parts)


def _cell_html(cell, assets, note_numbers: dict[str, int]) -> str:
    return "<br>".join(
        html
        for block in cell.blocks
        for html in [_cell_block_html(block, assets, note_numbers)]
        if html
    )


def _table_html(table, assets, note_numbers: dict[str, int]) -> str:
    rows: list[str] = []
    for row_index, row in enumerate(table.grid):
        cells: list[str] = []
        tag = "th" if row_index < table.header_rows else "td"
        for slot in row:
            if slot.kind != "origin" or slot.cell is None:
                continue
            attrs = ""
            if slot.cell.col_span > 1:
                attrs += f' colspan="{slot.cell.col_span}"'
            if slot.cell.row_span > 1:
                attrs += f' rowspan="{slot.cell.row_span}"'
            cells.append(f"<{tag}{attrs}>{_cell_html(slot.cell, assets, note_numbers)}</{tag}>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table>{''.join(rows)}</table>"


def _list_content(items, assets, note_numbers: dict[str, int], ilevel: int) -> tuple[list[dict], list[dict]]:
    """返回列表条目和需要提升到列表之后的图片块：office 列表条目只接受文本和嵌套列表。"""
    content: list[dict] = []
    hoisted_images: list[dict] = []
    for item in items:
        for block in item.blocks:
            if block.kind == "list":
                nested, nested_images = _list_block(block, assets, note_numbers, ilevel + 1)
                if nested:
                    content.append(nested)
                hoisted_images.extend(nested_images)
                continue
            for item_block in _inlines_to_blocks(block.content, assets, note_numbers):
                if item_block["type"] == "image":
                    hoisted_images.append(item_block)
                else:
                    content.append(item_block)
    return content, hoisted_images


def _list_block(block, assets, note_numbers: dict[str, int], ilevel: int) -> tuple[dict | None, list[dict]]:
    content, hoisted_images = _list_content(block.list.items, assets, note_numbers, ilevel)
    if not content:
        return None, hoisted_images
    list_block = {
        "type": "list",
        "attribute": "ordered" if block.list.marker in _MARKER_ORDERED else "unordered",
        "ilevel": ilevel,
        "start": block.list.start,
        "content": content,
    }
    return list_block, hoisted_images


def _block_to_office_blocks(
    block,
    assets,
    note_numbers: dict[str, int],
    linked_anchors: set[str],
) -> list[dict]:
    kind = block.kind
    if kind == "heading":
        extra = {"level": min(max(block.level or 1, 1), 6), "is_numbered_style": False}
        if block.anchor in linked_anchors:
            extra["anchor"] = block.anchor
        return _inlines_to_blocks(
            block.content, assets, note_numbers, block_type="title", extra=extra
        )
    if kind == "paragraph":
        return _inlines_to_blocks(block.content, assets, note_numbers)
    if kind == "math":
        return [{"type": "equation", "content": block.text or ""}]
    if kind == "code_block":
        return [{"type": "text", "content": block.text or ""}]
    if kind == "table":
        return [{"type": "table", "content": _table_html(block.table, assets, note_numbers)}]
    if kind == "list":
        list_block, hoisted_images = _list_block(block, assets, note_numbers, 0)
        return ([list_block] if list_block else []) + hoisted_images
    if kind == "block_quote":
        return [
            office_block
            for child in block.blocks or []
            for office_block in _block_to_office_blocks(
                child, assets, note_numbers, linked_anchors
            )
        ]
    return []


def _collect_linked_anchors(blocks) -> set[str]:
    """只保留文档内真的被链接引用的锚点，避免给每个标题写出无用的 <a id>。"""
    anchors: set[str] = set()
    for block in blocks:
        for inline in block.content or []:
            if inline.kind == "link" and inline.target and inline.target.kind == "anchor":
                anchors.add(inline.target.value)
        if block.kind == "list":
            anchors |= _collect_linked_anchors(
                child for item in block.list.items for child in item.blocks
            )
        elif block.kind == "block_quote":
            anchors |= _collect_linked_anchors(block.blocks or [])
        elif block.kind == "table":
            anchors |= _collect_linked_anchors(
                child
                for row in block.table.grid
                for slot in row
                if slot.kind == "origin" and slot.cell is not None
                for child in slot.cell.blocks
            )
    return anchors


def _notes_to_blocks(notes, assets, note_numbers: dict[str, int]) -> list[dict]:
    blocks: list[dict] = []
    for note in notes:
        body = " ".join(
            text
            for child in note.blocks
            for text in [_inlines_to_plain_text(child.content, assets, note_numbers)]
            if text
        )
        if body:
            blocks.append({"type": "text", "content": f"[{note_numbers[note.id]}] {body}"})
    return blocks


def document_to_page_blocks(document) -> list[dict]:
    """anydoc 文档模型没有分页概念，整篇文档产出一页 office 中间块。"""
    note_numbers = {note.id: index for index, note in enumerate(document.notes, start=1)}
    linked_anchors = _collect_linked_anchors(document.blocks)
    page_blocks = [
        office_block
        for block in document.blocks
        for office_block in _block_to_office_blocks(
            block, document.assets, note_numbers, linked_anchors
        )
    ]
    page_blocks.extend(_notes_to_blocks(document.notes, document.assets, note_numbers))
    return page_blocks
