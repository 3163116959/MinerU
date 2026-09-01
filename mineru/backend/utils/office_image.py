# Copyright (c) Opendatalab. All rights reserved.
import shutil
import subprocess
import tempfile
from functools import lru_cache
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Final

from PIL import Image, ImageChops, ImageDraw, ImageFont, UnidentifiedImageError
from loguru import logger

from mineru.utils.check_sys_env import is_windows_environment
from mineru.utils.pdf_reader import image_to_b64str


VECTOR_IMAGE_FORMATS = frozenset({"WMF", "EMF"})
VECTOR_IMAGE_EXTENSIONS = frozenset({".wmf", ".emf"})
VECTOR_IMAGE_CONTENT_TYPES = frozenset(
    {
        "image/x-wmf",
        "image/wmf",
        "image/x-emf",
        "image/emf",
        "application/x-msmetafile",
    }
)
PIL_IMAGE_LOAD_ERRORS = (UnidentifiedImageError, OSError, SyntaxError)
STANDARD_VECTOR_PLACEHOLDER_SIZE: Final = (320, 180)
STANDARD_VECTOR_PLACEHOLDER_LINES: Final = (
    "WMF/EMF placeholder",
    "Use Windows to parse",
    "the original image",
)
SOFFICE_TIMEOUT_SEC: Final = 120
# A4@300dpi。soffice 把矢量图等比缩放贴到 Draw 页面再导出，
# 只有保持页面自身纵横比才不会拉伸，导出后再裁掉页面白边。
VECTOR_RASTER_PAGE_SIZE: Final = (2480, 3508)


def is_vector_image(pil_image: Image.Image) -> bool:
    return (getattr(pil_image, "format", None) or "").upper() in VECTOR_IMAGE_FORMATS


def is_vector_image_part(
    part_name: object | None = None, content_type: str | None = None
) -> bool:
    suffix = PurePosixPath(str(part_name or "")).suffix.lower()
    if suffix in VECTOR_IMAGE_EXTENSIONS:
        return True
    normalized_content_type = (content_type or "").split(";", 1)[0].strip().lower()
    return normalized_content_type in VECTOR_IMAGE_CONTENT_TYPES


def _vector_image_format_label(
    part_name: object | None = None, content_type: str | None = None
) -> str:
    suffix = PurePosixPath(str(part_name or "")).suffix.lower()
    normalized_content_type = (content_type or "").lower()
    if suffix == ".wmf" or "wmf" in normalized_content_type:
        return "WMF"
    if suffix == ".emf" or "emf" in normalized_content_type:
        return "EMF"
    return "WMF/EMF"


def _load_placeholder_font(font_size: int) -> ImageFont.ImageFont:
    for font_name in (
        "DejaVuSans.ttf",
        "Arial.ttf",
        "LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_text_placeholder(
    size: tuple[int, int], lines: list[str]
) -> Image.Image:
    width = max(int(size[0]), 1)
    height = max(int(size[1]), 1)
    placeholder = Image.new("RGB", (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(placeholder)

    border_width = max(1, min(width, height) // 80)
    draw.rectangle(
        (0, 0, width - 1, height - 1),
        outline=(190, 190, 190),
        width=border_width,
    )

    max_text_width = max(width - 16, 1)
    max_text_height = max(height - 16, 1)
    fallback_text = "WMF/EMF"
    text = "\n".join(line for line in lines if line)
    if not text:
        text = fallback_text

    font = None
    spacing = 4
    bbox = None
    for font_size in range(max(min(width, height) // 7, 10), 7, -1):
        font = _load_placeholder_font(font_size)
        spacing = max(2, font_size // 4)
        bbox = draw.multiline_textbbox(
            (0, 0), text, font=font, spacing=spacing, align="center"
        )
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if text_width <= max_text_width and text_height <= max_text_height:
            break
    else:
        text = fallback_text
        font = _load_placeholder_font(max(min(width, height) // 5, 10))
        spacing = 2
        bbox = draw.multiline_textbbox(
            (0, 0), text, font=font, spacing=spacing, align="center"
        )

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    origin = ((width - text_width) / 2, (height - text_height) / 2)
    draw.multiline_text(
        origin,
        text,
        fill=(90, 90, 90),
        font=font,
        spacing=spacing,
        align="center",
    )
    return placeholder


@lru_cache(maxsize=1)
def _standard_vector_placeholder_data_uri() -> str:
    """生成并缓存标准 WMF/EMF 占位图，避免每张矢量图重复绘制。"""
    placeholder = create_text_placeholder(
        STANDARD_VECTOR_PLACEHOLDER_SIZE,
        list(STANDARD_VECTOR_PLACEHOLDER_LINES),
    )
    return image_to_b64str(placeholder, image_format="JPEG")


def get_standard_vector_placeholder_data_uri() -> str:
    """返回标准 WMF/EMF 占位图 data URI，供 Office 各格式复用。"""
    return _standard_vector_placeholder_data_uri()


@lru_cache(maxsize=1)
def _soffice_executable() -> str | None:
    """定位 LibreOffice 可执行文件；缺失只在首次告警一次。"""
    executable = shutil.which("soffice") or shutil.which("libreoffice")
    if executable is None:
        logger.warning(
            "soffice not found in PATH, WMF/EMF images fall back to placeholder. "
            "Install LibreOffice to rasterize them."
        )
    return executable


def _crop_page_margin(page: Image.Image) -> Image.Image | None:
    """裁掉 Draw 页面白边只留矢量图本体；整页全白说明源图为空。"""
    rgb = page.convert("RGB")
    bbox = ImageChops.difference(
        rgb, Image.new("RGB", rgb.size, (255, 255, 255))
    ).getbbox()
    return rgb.crop(bbox) if bbox else None


def rasterize_vector_image(image_data: bytes, image_format: str) -> str | None:
    """用 LibreOffice 把 WMF/EMF 光栅化为 PNG data URI；不可用或渲染为空返回 None。"""
    soffice = _soffice_executable()
    if soffice is None:
        return None

    width, height = VECTOR_RASTER_PAGE_SIZE
    suffix = ".emf" if image_format == "EMF" else ".wmf"
    with tempfile.TemporaryDirectory() as workdir:
        source = Path(workdir) / f"vector{suffix}"
        source.write_bytes(image_data)
        outdir = Path(workdir) / "out"
        try:
            subprocess.run(
                [
                    soffice,
                    "--headless",
                    "--norestore",
                    # 每次用独立 profile：并发转换不会抢同一把 LibreOffice 用户目录锁
                    f"-env:UserInstallation=file://{workdir}/profile",
                    "--convert-to",
                    f'png:draw_png_Export:{{"PixelWidth":{{"type":"long","value":{width}}},'
                    f'"PixelHeight":{{"type":"long","value":{height}}}}}',
                    "--outdir",
                    str(outdir),
                    str(source),
                ],
                check=True,
                capture_output=True,
                timeout=SOFFICE_TIMEOUT_SEC,
            )
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning(f"LibreOffice failed to rasterize {image_format} image: {e}")
            return None

        rendered = outdir / f"{source.stem}.png"
        if not rendered.exists():
            logger.warning(f"LibreOffice produced no PNG for {image_format} image")
            return None

        with Image.open(rendered) as page:
            cropped = _crop_page_margin(page)

    if cropped is None:
        logger.warning(f"LibreOffice rendered a blank page for {image_format} image")
        return None

    return image_to_b64str(cropped, image_format="PNG")


def serialize_vector_image_with_placeholder(
    pil_image: Image.Image,
    image_format_override: str | None = None,
    image_bytes: bytes | None = None,
) -> str:
    image_format = (
        image_format_override or getattr(pil_image, "format", None) or "WMF/EMF"
    ).upper()

    # Windows 上 Pillow 走系统 GDI 能直接渲染；其他平台没有 handler，交给 LibreOffice。
    if is_windows_environment():
        try:
            pil_image.load()
            return image_to_b64str(pil_image, image_format="PNG")
        except PIL_IMAGE_LOAD_ERRORS as e:
            logger.warning(
                f"Failed to render {image_format} image: {e}, size: {pil_image.size}"
            )

    if image_bytes is not None:
        rendered = rasterize_vector_image(image_bytes, image_format)
        if rendered is not None:
            return rendered

    logger.warning(
        f"Using placeholder for {image_format} image, size: {pil_image.size}"
    )
    return get_standard_vector_placeholder_data_uri()


def serialize_vector_part_with_placeholder(
    part_name: object | None = None,
    content_type: str | None = None,
    size: tuple[int, int] = (320, 180),
) -> str:
    image_format = _vector_image_format_label(part_name, content_type)
    logger.warning(
        f"Skipping {image_format} image part before Pillow load, "
        f"part_name={part_name}, content_type={content_type}, requested_size={size}"
    )
    return get_standard_vector_placeholder_data_uri()


def serialize_office_image(
    image_data: bytes,
    *,
    part_name: object | None = None,
    content_type: str | None = None,
) -> str | None:
    if is_vector_image_part(part_name, content_type):
        image_format = _vector_image_format_label(part_name, content_type)
        try:
            pil_image = Image.open(BytesIO(image_data))
        except PIL_IMAGE_LOAD_ERRORS as e:
            # Pillow 认不出头部不代表 LibreOffice 渲染不了，先试转换再退占位图。
            logger.warning(
                f"Warning: vector image cannot be opened by Pillow: {e}, "
                f"part_name={part_name}, content_type={content_type}"
            )
            return rasterize_vector_image(
                image_data, image_format
            ) or serialize_vector_part_with_placeholder(part_name, content_type)

        return serialize_vector_image_with_placeholder(
            pil_image,
            image_format_override=image_format,
            image_bytes=image_data,
        )

    try:
        pil_image = Image.open(BytesIO(image_data))
        pil_image.load()
    except PIL_IMAGE_LOAD_ERRORS as e:
        logger.warning(
            f"Warning: image cannot be loaded by Pillow: {e}, "
            f"part_name={part_name}, content_type={content_type}"
        )
        return None

    if is_vector_image(pil_image):
        return serialize_vector_image_with_placeholder(
            pil_image, image_bytes=image_data
        )

    if pil_image.mode == "RGB":
        return image_to_b64str(pil_image, image_format="JPEG")

    if pil_image.mode in {"RGBA", "LA"} or (
        pil_image.mode == "P" and "transparency" in pil_image.info
    ):
        return image_to_b64str(pil_image.convert("RGBA"), image_format="PNG")

    return image_to_b64str(pil_image.convert("RGB"), image_format="JPEG")
