# Copyright (c) Opendatalab. All rights reserved.
"""把 anydoc 的 Markdown 输出转成 office 中间块。

anydoc 的 PDF 转换只产出 Markdown（没有文档模型），所以纯文本 PDF 走这条路径，
再复用 office 中间层生成 middle_json / content_list，保证与其他后端输出一致。
"""
import re
from html import escape

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE_RE = re.compile(r"^\s*(```|~~~)(.*)$")
MATH_FENCE = "$$"
LIST_ITEM_RE = re.compile(r"^(\s*)(?:([-*+])|(\d+)[.)])\s+(.*)$")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$")
INLINE_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]*)\)"
    r"|\[(?P<label>[^\]]*)\]\((?P<url>[^)]*)\)"
    r"|\$(?P<math>[^$\n]+)\$"
    r"|\*\*(?P<bold>.+?)\*\*|__(?P<bold2>.+?)__"
    r"|~~(?P<strike>.+?)~~"
    r"|\*(?P<italic>[^*\n]+?)\*|_(?P<italic2>[^_\n]+?)_"
    r"|`(?P<code>[^`\n]+?)`"
)
_INDENT_WIDTH = 2


def _styled(content: str, style: str) -> str:
    if not content:
        return ""
    return f'<text style="{style}">{content}</text>'


def render_inline(text: str) -> str:
    """把 Markdown 行内语法转成 office content 标签，未识别部分保持原文。"""
    parts: list[str] = []
    position = 0
    for match in INLINE_RE.finditer(text):
        parts.append(text[position:match.start()])
        position = match.end()
        groups = match.groupdict()
        if groups["src"] is not None:
            parts.append(groups["alt"])
        elif groups["url"] is not None:
            label = render_inline(groups["label"])
            if "<text" not in label:
                label = f"<text>{label}</text>"
            parts.append(f"<hyperlink>{label}<url>{groups['url']}</url></hyperlink>")
        elif groups["math"] is not None:
            parts.append(f"<eq>{groups['math']}</eq>")
        elif groups["bold"] is not None or groups["bold2"] is not None:
            parts.append(_styled(render_inline(groups["bold"] or groups["bold2"]), "bold"))
        elif groups["strike"] is not None:
            parts.append(_styled(render_inline(groups["strike"]), "strikethrough"))
        elif groups["italic"] is not None or groups["italic2"] is not None:
            parts.append(_styled(render_inline(groups["italic"] or groups["italic2"]), "italic"))
        elif groups["code"] is not None:
            parts.append(groups["code"])
    parts.append(text[position:])
    return "".join(parts)


def _plain_text(text: str) -> str:
    """表格单元格只保留纯文本：HTML 单元格里渲染不了 Markdown 行内语法。"""
    return escape(re.sub(r"<[^>]+>", "", render_inline(text)))


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def _parse_table(lines: list[str], index: int) -> tuple[dict, int]:
    header = _split_table_row(lines[index])
    cursor = index + 2
    rows = []
    while cursor < len(lines) and lines[cursor].lstrip().startswith("|"):
        rows.append(_split_table_row(lines[cursor]))
        cursor += 1

    head_html = "".join(f"<th>{_plain_text(cell)}</th>" for cell in header)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{_plain_text(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    table_html = f"<table><tr>{head_html}</tr>{body_html}</table>"
    return {"type": "table", "content": table_html}, cursor


def _is_table_start(lines: list[str], index: int) -> bool:
    return (
        lines[index].lstrip().startswith("|")
        and index + 1 < len(lines)
        and TABLE_SEPARATOR_RE.match(lines[index + 1])
        and "-" in lines[index + 1]
    )


def _parse_fenced_code(lines: list[str], index: int) -> tuple[dict | None, int]:
    fence = FENCE_RE.match(lines[index]).group(1)
    cursor = index + 1
    body: list[str] = []
    while cursor < len(lines) and not lines[cursor].lstrip().startswith(fence):
        body.append(lines[cursor])
        cursor += 1
    content = "\n".join(body).strip()
    block = {"type": "text", "content": content} if content else None
    return block, cursor + 1


def _parse_math_block(lines: list[str], index: int) -> tuple[dict | None, int]:
    cursor = index + 1
    body: list[str] = []
    while cursor < len(lines) and lines[cursor].strip() != MATH_FENCE:
        body.append(lines[cursor])
        cursor += 1
    content = "\n".join(body).strip()
    block = {"type": "equation", "content": content} if content else None
    return block, cursor + 1


def _list_item_depth(indent: str) -> int:
    return len(indent.expandtabs(4)) // _INDENT_WIDTH


def _new_list_block(ordered: bool, start: int, ilevel: int) -> dict:
    return {
        "type": "list",
        "attribute": "ordered" if ordered else "unordered",
        "ilevel": ilevel,
        "start": start,
        "content": [],
    }


def _parse_list(lines: list[str], index: int) -> tuple[dict, int]:
    """按缩进宽度还原嵌套列表；同级条目挂在同一个列表块下。"""
    root_match = LIST_ITEM_RE.match(lines[index])
    root_depth = _list_item_depth(root_match.group(1))
    root = _new_list_block(
        ordered=root_match.group(3) is not None,
        start=int(root_match.group(3)) if root_match.group(3) else 1,
        ilevel=0,
    )
    stack: list[tuple[int, dict]] = [(root_depth, root)]
    cursor = index

    while cursor < len(lines):
        match = LIST_ITEM_RE.match(lines[cursor])
        if not match:
            break
        depth = _list_item_depth(match.group(1))
        ordered = match.group(3) is not None

        while len(stack) > 1 and depth < stack[-1][0]:
            stack.pop()

        if depth > stack[-1][0]:
            nested = _new_list_block(
                ordered=ordered,
                start=int(match.group(3)) if ordered else 1,
                ilevel=len(stack),
            )
            stack[-1][1]["content"].append(nested)
            stack.append((depth, nested))

        stack[-1][1]["content"].append(
            {"type": "text", "content": render_inline(match.group(4).strip())}
        )
        cursor += 1

    return root, cursor


def _parse_paragraph(lines: list[str], index: int) -> tuple[dict | None, int]:
    cursor = index
    body: list[str] = []
    while cursor < len(lines):
        line = lines[cursor]
        if not line.strip() or HEADING_RE.match(line) or LIST_ITEM_RE.match(line):
            break
        if FENCE_RE.match(line) or line.strip() == MATH_FENCE or _is_table_start(lines, cursor):
            break
        body.append(line.strip())
        cursor += 1
    content = render_inline(" ".join(body).strip())
    block = {"type": "text", "content": content} if content else None
    return block, cursor


def markdown_to_page_blocks(markdown: str) -> list[dict]:
    lines = markdown.splitlines()
    page_blocks: list[dict] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue

        heading_match = HEADING_RE.match(line)
        if heading_match:
            content = render_inline(heading_match.group(2).strip())
            if content:
                page_blocks.append(
                    {
                        "type": "title",
                        "content": content,
                        "level": len(heading_match.group(1)),
                        "is_numbered_style": False,
                    }
                )
            index += 1
            continue

        if FENCE_RE.match(line):
            block, index = _parse_fenced_code(lines, index)
        elif line.strip() == MATH_FENCE:
            block, index = _parse_math_block(lines, index)
        elif _is_table_start(lines, index):
            block, index = _parse_table(lines, index)
        elif LIST_ITEM_RE.match(line):
            block, index = _parse_list(lines, index)
        else:
            block, index = _parse_paragraph(lines, index)

        if block:
            page_blocks.append(block)

    return page_blocks
