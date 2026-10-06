# Copyright (c) Opendatalab. All rights reserved.
import asyncio
import json
import os
from pathlib import Path
from typing import NamedTuple, Sequence

from loguru import logger

from mineru.cli.output_paths import build_parse_dir
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_bytes
from mineru.utils.pdf_image_tools import images_bytes_to_pdf_bytes
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.backend.office.office_middle_json_mkcontent import union_make as office_union_make
from mineru.backend.office.image_analyze import aio_analyze_office_images
from mineru.backend.anydoc.anydoc_analyze import anydoc_analyze
from mineru.utils.pdfium_guard import (
    get_loadable_pdfium_page_indices,
    rewrite_pdf_bytes_with_pdfium,
)

os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
if os.getenv("MINERU_LMDEPLOY_DEVICE", "") == "maca":
    import torch
    torch.backends.cudnn.enabled = False


pdf_suffixes = ["pdf"]
image_suffixes = ["png", "jpeg", "jp2", "webp", "gif", "bmp", "jpg", "tiff"]
# office 文档一律由 anydoc 解析。
office_suffixes = [
    "docx", "pptx", "xlsx", "doc", "ppt", "xls",
    "odt", "ods", "odp", "rtf", "epub", "csv",
]
# 音频一律由 whisperx 解析；取值为 magika 标签，m4a/aac 容器被识别为 mp4。
audio_suffixes = ["wav", "mp3", "flac", "ogg", "wma", "mp4"]

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Maximum UTF-8 byte length allowed for task stems used in filenames.
# 200 bytes is chosen to stay well below common filesystem limits (e.g. 255 bytes)
# and to prevent generating excessively long or incompatible filenames.
MAX_TASK_STEM_BYTES = 200
# 仅支持 hybrid-http-client：VLM 走远程 OpenAI 兼容服务。
VLM_BACKEND = "http-client"


def utf8_byte_length(value: str) -> int:
    return len(value.encode("utf-8"))


def truncate_to_utf8_bytes(value: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""

    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value

    truncated = encoded[:max_bytes]
    while truncated:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError as exc:
            truncated = truncated[:exc.start]
    return ""


def normalize_task_stem(stem: str, max_bytes: int = MAX_TASK_STEM_BYTES) -> str:
    return truncate_to_utf8_bytes(stem, max_bytes)


def normalize_upload_filename(upload_name: str) -> str:
    sanitized_name = Path(upload_name).name
    sanitized_path = Path(sanitized_name)
    normalized_stem = normalize_task_stem(sanitized_path.stem)
    return f"{normalized_stem}{sanitized_path.suffix}"


def build_task_stem_candidate(
    stem: str,
    suffix: str = "",
    max_bytes: int = MAX_TASK_STEM_BYTES,
) -> str:
    if utf8_byte_length(f"{stem}{suffix}") <= max_bytes:
        return f"{stem}{suffix}"
    suffix_bytes = utf8_byte_length(suffix)
    if suffix_bytes >= max_bytes:
        return truncate_to_utf8_bytes(suffix, max_bytes)
    return f"{truncate_to_utf8_bytes(stem, max_bytes - suffix_bytes)}{suffix}"


def uniquify_task_stems(
    stems: Sequence[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Assign task-local unique stems while preserving input order."""
    normalized_inputs = [normalize_task_stem(stem) for stem in stems]
    raw_keys = {stem.casefold() for stem in normalized_inputs}
    occurrence_counts: dict[str, int] = {}
    assigned_keys: set[str] = set()
    unique_stems: list[str] = []
    renamed: list[tuple[str, str]] = []

    for stem, normalized_stem in zip(stems, normalized_inputs):
        stem_base = normalized_stem or stem
        stem_key = stem_base.casefold()
        seen_count = occurrence_counts.get(stem_key, 0)
        occurrence_counts[stem_key] = seen_count + 1

        if seen_count == 0 and stem_key not in assigned_keys:
            effective_stem = stem_base
        else:
            suffix = seen_count + 1
            while True:
                candidate = build_task_stem_candidate(stem_base, f"_{suffix}")
                candidate_key = candidate.casefold()
                if candidate_key not in raw_keys and candidate_key not in assigned_keys:
                    effective_stem = candidate
                    break
                suffix += 1

        assigned_keys.add(effective_stem.casefold())
        unique_stems.append(effective_stem)
        if effective_stem != stem:
            renamed.append((stem, effective_stem))

    return unique_stems, renamed


def read_fn(path, file_suffix: str | None = None):
    if not isinstance(path, Path):
        path = Path(path)
    with open(str(path), "rb") as input_file:
        file_bytes = input_file.read()
        if file_suffix is None:
            file_suffix = guess_suffix_by_bytes(file_bytes, path)
        if file_suffix in image_suffixes:
            return images_bytes_to_pdf_bytes(file_bytes)
        elif file_suffix in pdf_suffixes + office_suffixes + audio_suffixes:
            return file_bytes
        else:
            raise Exception(f"Unknown file suffix: {file_suffix}")


def prepare_env(output_dir, pdf_file_name, parse_method):
    local_md_dir = str(os.path.join(output_dir, pdf_file_name, parse_method))
    local_image_dir = os.path.join(str(local_md_dir), "images")
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    return local_image_dir, local_md_dir


def convert_pdf_bytes_to_bytes(pdf_bytes, start_page_id=0, end_page_id=None):
    try:
        rebuilt_pdf_bytes = rewrite_pdf_bytes_with_pdfium(
            pdf_bytes,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
        )
        if rebuilt_pdf_bytes:
            return rebuilt_pdf_bytes
        logger.warning(
            "PDFium rewrite returned empty bytes, trying to skip broken pages."
        )
    except Exception as fallback_error:
        logger.warning(
            f"Error in converting PDF bytes with pdfium: {fallback_error}, "
            "trying to skip broken pages."
        )

    try:
        loadable_page_indices, broken_page_indices = get_loadable_pdfium_page_indices(
            pdf_bytes,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
        )
        if broken_page_indices:
            skipped_pages = [page_index + 1 for page_index in broken_page_indices]
            logger.warning(
                f"Skipped broken PDF pages during PDFium rewrite: {skipped_pages}"
            )
        if not loadable_page_indices:
            logger.warning(
                "PDFium skip-broken-page rewrite found no loadable pages, "
                "using original PDF bytes."
            )
            return pdf_bytes

        rebuilt_pdf_bytes = rewrite_pdf_bytes_with_pdfium(
            pdf_bytes,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            page_indices=loadable_page_indices,
        )
        if rebuilt_pdf_bytes:
            return rebuilt_pdf_bytes
        logger.warning(
            "PDFium skip-broken-page rewrite returned empty bytes, "
            "using original PDF bytes."
        )
    except Exception as fallback_error:
        logger.warning(
            "Error in converting PDF bytes with skip-broken-page fallback: "
            f"{fallback_error}, using original PDF bytes."
        )
    return pdf_bytes


def _process_output(
        pdf_info,
        pdf_bytes,
        pdf_file_name,
        local_md_dir,
        local_image_dir,
        md_writer,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_orig_pdf,
        f_dump_md,
        f_dump_content_list,
        f_dump_middle_json,
        f_dump_model_output,
        f_make_md_mode,
        middle_json,
        model_output=None,
        process_mode="vlm",
        orig_file_suffix="pdf",
):
    if process_mode == "vlm":
        make_func = vlm_union_make
    elif process_mode == "office":
        make_func = office_union_make
    else:
        raise Exception(f"Unknown process_mode: {process_mode}")
    """处理输出文件"""
    if f_draw_layout_bbox:
        try:
            draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")
        except Exception as exc:
            logger.warning(f"Skipping layout bbox visualization for {pdf_file_name}: {exc}")

    if f_draw_span_bbox:
        try:
            draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")
        except Exception as exc:
            logger.warning(f"Skipping span bbox visualization for {pdf_file_name}: {exc}")

    if f_dump_orig_pdf:
        md_writer.write(
            f"{pdf_file_name}_origin.{orig_file_suffix}",
            pdf_bytes,
        )

    image_dir = str(os.path.basename(local_image_dir))

    if f_dump_md:
        md_content_str = make_func(pdf_info, f_make_md_mode, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}.md",
            md_content_str,
        )

    if f_dump_content_list:

        content_list = make_func(pdf_info, MakeMode.CONTENT_LIST, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )

        content_list_v2 = make_func(pdf_info, MakeMode.CONTENT_LIST_V2, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}_content_list_v2.json",
            json.dumps(content_list_v2, ensure_ascii=False, indent=4),
        )


    if f_dump_middle_json:
        md_writer.write_string(
            f"{pdf_file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4),
        )

    if f_dump_model_output:
        md_writer.write_string(
            f"{pdf_file_name}_model.json",
            json.dumps(model_output, ensure_ascii=False, indent=4),
        )

    logger.debug(f"local output dir is {local_md_dir}")


async def _async_process_hybrid(
        output_dir,
        pdf_file_names,
        pdf_bytes_list,
        parse_method,
        f_dump_md,
        f_dump_middle_json,
        f_dump_orig_pdf,
        server_url,
        **kwargs,
):
    # 延迟导入：hybrid 依赖 torch 等重模块，不拖慢服务启动。
    from mineru.backend.hybrid.hybrid_analyze import aio_doc_analyze as aio_hybrid_doc_analyze

    for pdf_file_name, pdf_bytes in zip(pdf_file_names, pdf_bytes_list):
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, f"hybrid_{parse_method}")
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

        middle_json, infer_result = await aio_hybrid_doc_analyze(
            pdf_bytes,
            image_writer=image_writer,
            backend=VLM_BACKEND,
            parse_method=parse_method,
            inline_formula_enable=True,
            server_url=server_url,
            image_analysis=True,
            **kwargs,
        )

        _process_output(
            middle_json["pdf_info"], pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
            md_writer, False, False, f_dump_orig_pdf,
            f_dump_md, False, f_dump_middle_json, False,
            MakeMode.MM_MD, middle_json, infer_result, process_mode="vlm"
        )


class _ParsedDoc(NamedTuple):
    index: int
    file_name: str
    file_bytes: bytes
    file_suffix: str
    local_image_dir: str
    local_md_dir: str
    md_writer: FileBasedDataWriter
    middle_json: dict
    infer_result: list


def _parse_office_docs(
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
) -> list[_ParsedDoc]:
    """只做解析，不落盘：图片分析要在写出 md/content_list 之前插入 caption。"""
    parsed: list[_ParsedDoc] = []
    for i, file_bytes in enumerate(pdf_bytes_list):
        file_suffix = guess_suffix_by_bytes(file_bytes)
        if file_suffix not in office_suffixes:
            continue

        pdf_file_name = pdf_file_names[i]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, "office")
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

        middle_json, infer_result = anydoc_analyze(
            file_bytes,
            file_suffix,
            image_writer=image_writer,
        )
        parsed.append(
            _ParsedDoc(
                i, pdf_file_name, file_bytes, file_suffix,
                local_image_dir, local_md_dir, md_writer, middle_json, infer_result,
            )
        )
    return parsed


def _write_parsed_outputs(
        parsed_list: list[_ParsedDoc],
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=True,
        f_dump_orig_file=True,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.MM_MD,
) -> list[int]:
    for item in parsed_list:
        _process_output(
            item.middle_json["pdf_info"], item.file_bytes, item.file_name,
            item.local_md_dir, item.local_image_dir, item.md_writer,
            False, False, f_dump_orig_file,
            f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
            f_make_md_mode, item.middle_json, item.infer_result,
            process_mode="office", orig_file_suffix=item.file_suffix,
        )
    return [item.index for item in parsed_list]


def _parse_audio_docs(
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=True,
        f_dump_orig_file=True,
        f_dump_content_list=True,
) -> list[int]:
    """解析并落盘音频，返回已处理的下标；音频无版面信息，不走 _process_output。"""
    from mineru.backend.audio.audio_analyze import audio_analyze
    from mineru.backend.audio.audio_middle_json_mkcontent import (
        make_content_list,
        make_markdown,
    )

    handled: list[int] = []
    for i, file_bytes in enumerate(pdf_bytes_list):
        file_suffix = guess_suffix_by_bytes(file_bytes)
        if file_suffix not in audio_suffixes:
            continue

        file_name = pdf_file_names[i]
        # 音频没有图片产物，不用 prepare_env 以免生成空 images 目录。
        local_md_dir = str(build_parse_dir(output_dir, file_name, "", "", is_audio=True))
        os.makedirs(local_md_dir, exist_ok=True)
        md_writer = FileBasedDataWriter(local_md_dir)
        middle_json, results = audio_analyze(file_bytes, file_suffix)

        if f_dump_orig_file:
            md_writer.write(f"{file_name}_origin.{file_suffix}", file_bytes)
        if f_dump_md:
            md_writer.write_string(f"{file_name}.md", make_markdown(middle_json))
        if f_dump_content_list:
            md_writer.write_string(
                f"{file_name}_content_list.json",
                json.dumps(make_content_list(middle_json), ensure_ascii=False, indent=4),
            )
        if f_dump_middle_json:
            md_writer.write_string(
                f"{file_name}_middle.json",
                json.dumps(middle_json, ensure_ascii=False, indent=4),
            )
        if f_dump_model_output:
            md_writer.write_string(
                f"{file_name}_model.json",
                json.dumps(results, ensure_ascii=False, indent=4),
            )
        logger.debug(f"local output dir is {local_md_dir}")
        handled.append(i)
    return handled


def _remove_handled(indices, pdf_bytes_list, pdf_file_names, p_lang_list):
    for index in sorted(indices, reverse=True):
        del pdf_bytes_list[index]
        del pdf_file_names[index]
        del p_lang_list[index]


async def aio_do_parse(
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        p_lang_list: list[str],
        parse_method: str,
        server_url=None,
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_orig_pdf=True,
        **kwargs,
):
    # 音频推理同步且耗时，同 office 一样放到线程中。
    audio_handled = await asyncio.to_thread(
        _parse_audio_docs, output_dir, pdf_file_names, pdf_bytes_list,
        f_dump_md=f_dump_md,
        f_dump_middle_json=f_dump_middle_json,
        f_dump_model_output=False,
        f_dump_orig_file=f_dump_orig_pdf,
        f_dump_content_list=False,
    )
    _remove_handled(audio_handled, pdf_bytes_list, pdf_file_names, p_lang_list)
    # Office 解析是同步且可能耗时的操作，异步入口需要放到线程中避免阻塞事件循环。
    office_parsed = await asyncio.to_thread(
        _parse_office_docs, output_dir, pdf_file_names, pdf_bytes_list
    )
    # 图片分析必须留在事件循环里：async 引擎不支持同步 predict。
    for item in office_parsed:
        await aio_analyze_office_images(
            item.middle_json,
            item.local_image_dir,
            vlm_backend=VLM_BACKEND,
            server_url=server_url,
            **kwargs,
        )
    need_remove_index = await asyncio.to_thread(
        _write_parsed_outputs,
        office_parsed,
        f_dump_md=f_dump_md,
        f_dump_middle_json=f_dump_middle_json,
        f_dump_model_output=False,
        f_dump_orig_file=f_dump_orig_pdf,
        f_dump_content_list=False,
    )
    _remove_handled(need_remove_index, pdf_bytes_list, pdf_file_names, p_lang_list)
    if not pdf_bytes_list:
        logger.info("No PDF or image files left to process.")
        return

    # 统一经 pdfium 重写，跳过损坏页，避免下游解析失败。
    pdf_bytes_list = [convert_pdf_bytes_to_bytes(pdf_bytes) for pdf_bytes in pdf_bytes_list]

    os.environ['MINERU_VLM_TABLE_ENABLE'] = "True"
    os.environ['MINERU_VLM_FORMULA_ENABLE'] = "true"

    await _async_process_hybrid(
        output_dir, pdf_file_names, pdf_bytes_list, parse_method,
        f_dump_md, f_dump_middle_json, f_dump_orig_pdf, server_url, **kwargs,
    )
