# Copyright (c) Opendatalab. All rights reserved.
from dataclasses import dataclass
from typing import Annotated, Any, Optional

from fastapi import File, Form, Request, UploadFile

from mineru.cli.public_http_client_policy import validate_public_http_client_request

# kb-demo 只用 hybrid-http-client + auto 判定 + 中文 OCR，对外不再开放选择。
PARSE_BACKEND = "hybrid-http-client"
PARSE_METHOD = "auto"
PARSE_LANG = "ch"
SWAGGER_UI_FILE_ARRAY_SCHEMA_EXTRA = {
    # Swagger UI 5 currently fails to render a usable multi-file picker when
    # FastAPI emits OpenAPI 3.1 byte arrays with contentMediaType.
    "items": {"type": "string", "format": "binary"}
}


@dataclass
class ParseRequestOptions:
    """保存公开解析接口的表单参数。"""

    files: list[UploadFile]
    server_url: Optional[str]
    model: Optional[str]
    api_key: Optional[str]
    return_md: bool
    return_middle_json: bool
    return_images: bool
    response_format_zip: bool
    return_original_file: bool


def build_vlm_client_kwargs(
    model: Optional[str],
    api_key: Optional[str],
) -> dict[str, Any]:
    """把请求级模型信息转换成 VLM http-client 的初始化参数。"""
    client_kwargs: dict[str, Any] = {}
    if model:
        client_kwargs["model_name"] = model
    if api_key:
        client_kwargs["server_headers"] = {"Authorization": f"Bearer {api_key}"}
    return client_kwargs


async def parse_request_form(
    request: Request,
    files: Annotated[
        list[UploadFile],
        File(
            description=(
                "Upload files for parsing. Supported: PDF; images (png/jpg/jpeg/jp2/webp/gif/bmp/tiff); "
                "office (docx/pptx/xlsx/doc/ppt/xls/odt/ods/odp/rtf/epub/csv); "
                "audio (wav/mp3/flac/ogg/wma/m4a/aac, and mp4 whose audio track is transcribed). "
                "File type is detected from content, not the extension."
            ),
            json_schema_extra=SWAGGER_UI_FILE_ARRAY_SCHEMA_EXTRA,
        ),
    ],
    server_url: Annotated[
        Optional[str],
        Form(
            description="(Adapted only for <vlm/hybrid>-http-client backend)openai compatible server url, e.g., http://127.0.0.1:30000",
        ),
    ] = None,
    model: Annotated[
        Optional[str],
        Form(
            description=(
                "(Adapted only for <vlm/hybrid>-http-client backend)model name served by "
                "server_url, e.g., MinerU2.5-2509-1.2B. Defaults to the only model exposed "
                "by the server."
            ),
        ),
    ] = None,
    api_key: Annotated[
        Optional[str],
        Form(
            description=(
                "(Adapted only for <vlm/hybrid>-http-client backend)API key sent to "
                "server_url as `Authorization: Bearer <api_key>`."
            ),
        ),
    ] = None,
    return_md: Annotated[
        bool,
        Form(description="Return markdown content in response. Audio: speaker-labelled transcript with timestamps"),
    ] = True,
    return_middle_json: Annotated[
        bool,
        Form(description="Return middle JSON in response. Audio: processed segments with speaker and word-level timestamps"),
    ] = False,
    return_images: Annotated[
        bool,
        Form(description="Return extracted images in response. Audio has no images"),
    ] = False,
    response_format_zip: Annotated[
        bool,
        Form(description="Return results as a ZIP file instead of JSON"),
    ] = False,
    return_original_file: Annotated[
        bool,
        Form(
            description=(
                "Include the processed original input file in the ZIP result; "
                "ignored unless response_format_zip=true"
            ),
        ),
    ] = False,
) -> ParseRequestOptions:
    """解析公开 multipart 表单。"""
    validate_public_http_client_request(
        public_bind_exposed=bool(
            getattr(request.app.state, "public_bind_exposed", False)
        ),
        allow_public_http_client=bool(
            getattr(request.app.state, "allow_public_http_client", False)
        ),
        backend=PARSE_BACKEND,
        server_url=server_url,
    )
    return ParseRequestOptions(
        files=files,
        server_url=server_url,
        model=model,
        api_key=api_key,
        return_md=return_md,
        return_middle_json=return_middle_json,
        return_images=return_images,
        response_format_zip=response_format_zip,
        return_original_file=return_original_file and response_format_zip,
    )
