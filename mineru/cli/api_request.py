# Copyright (c) Opendatalab. All rights reserved.
from dataclasses import dataclass
from typing import Annotated, Any, Optional

from fastapi import File, Form, HTTPException, Request, UploadFile

from mineru.cli.backend_options import (
    BACKEND_SCHEMA_EXTRA,
    DEFAULT_BACKEND,
    DEFAULT_HYBRID_EFFORT,
    HYBRID_EFFORT_SCHEMA_EXTRA,
    validate_backend as validate_public_backend,
    validate_effort as validate_public_effort,
)
from mineru.cli.public_http_client_policy import validate_public_http_client_request
from mineru.utils.ocr_language import (
    PUBLIC_OCR_LANGUAGE_SCHEMA_EXTRA,
    format_public_ocr_lang_description,
    validate_public_ocr_lang_list,
)

ALLOWED_PARSE_METHODS = {"auto", "txt", "ocr"}
SWAGGER_UI_FILE_ARRAY_SCHEMA_EXTRA = {
    # Swagger UI 5 currently fails to render a usable multi-file picker when
    # FastAPI emits OpenAPI 3.1 byte arrays with contentMediaType.
    "items": {"type": "string", "format": "binary"}
}


@dataclass
class ParseRequestOptions:
    """保存公开解析接口共用的表单参数，供 API 与 Router 复用。"""

    files: list[UploadFile]
    lang_list: list[str]
    backend: str
    effort: str
    parse_method: str
    formula_enable: bool
    table_enable: bool
    image_analysis: bool
    server_url: Optional[str]
    model: Optional[str]
    api_key: Optional[str]
    return_md: bool
    return_middle_json: bool
    return_model_output: bool
    return_content_list: bool
    return_images: bool
    response_format_zip: bool
    return_original_file: bool
    client_side_output_generation: bool
    start_page_id: int
    end_page_id: int


def validate_parse_method(parse_method: str) -> str:
    """校验公开 API 允许的 PDF 解析方式，避免各入口维护不同规则。"""
    if parse_method not in ALLOWED_PARSE_METHODS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid parse_method. Allowed values: "
                + ", ".join(sorted(ALLOWED_PARSE_METHODS))
            ),
        )
    return parse_method


def validate_parse_backend(backend: str) -> str:
    """校验公开 API 允许的解析后端，避免旧入口名进入下游执行链路。"""
    try:
        return validate_public_backend(backend)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def validate_parse_effort(effort: str) -> str:
    """校验公开 API 允许的 hybrid effort，避免非法值进入解析链路。"""
    try:
        return validate_public_effort(effort)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def validate_parse_model_credentials(
    backend: str,
    model: Optional[str],
    api_key: Optional[str],
) -> None:
    """model/api_key 只对 http-client 后端有意义，其他后端传入即报错。"""
    if backend.endswith("-http-client"):
        return
    for name, value in (("model", model), ("api_key", api_key)):
        if value:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{name} is only supported by <vlm/hybrid>-http-client backends, "
                    f"but backend is '{backend}'."
                ),
            )


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


def validate_parse_lang_list(lang_list: list[str]) -> list[str]:
    """校验公开 API 允许的 OCR 语言列表，避免旧语言入口进入解析链路。"""
    try:
        return validate_public_ocr_lang_list(lang_list)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def parse_request_form(
    request: Request,
    files: Annotated[
        list[UploadFile],
        File(
            description="Upload PDF, image, DOCX, PPTX, or XLSX files for parsing",
            json_schema_extra=SWAGGER_UI_FILE_ARRAY_SCHEMA_EXTRA,
        ),
    ],
    lang_list: Annotated[
        list[str],
        Form(
            description=format_public_ocr_lang_description(),
            json_schema_extra=PUBLIC_OCR_LANGUAGE_SCHEMA_EXTRA,
        ),
    ] = ["ch"],
    backend: Annotated[
        str,
        Form(
            description="""The backend for parsing:
- pipeline: More general, supports multiple languages, hallucination-free.
- vlm-engine: High accuracy via local computing power, supports Chinese and English documents only.
- vlm-http-client: High accuracy via remote computing power(client suitable for openai-compatible servers), supports Chinese and English documents only.
- hybrid-engine: Hybrid parsing via local computing power, supports multiple languages. Use effort to switch medium/high behavior.
- hybrid-http-client: Hybrid parsing via remote computing power but requires a little local computing power(client suitable for openai-compatible servers), supports multiple languages. Use effort to switch medium/high behavior.""",
            json_schema_extra=BACKEND_SCHEMA_EXTRA,
        ),
    ] = DEFAULT_BACKEND,
    effort: Annotated[
        str,
        Form(
            description="""(Adapted only for hybrid backend) Hybrid parsing effort:
- medium: Faster parsing for most documents, balancing accuracy and efficiency. Image/chart analysis is disabled.
- high: Higher-accuracy parsing with image/chart analysis support, which may take longer.""",
            json_schema_extra=HYBRID_EFFORT_SCHEMA_EXTRA,
        ),
    ] = DEFAULT_HYBRID_EFFORT,
    parse_method: Annotated[
        str,
        Form(
            description="""(Adapted only for pipeline and hybrid backend)The method for parsing PDF:
- auto: Automatically determine the method based on the file type
- txt: Use text extraction method
- ocr: Use OCR method for image-based PDFs
""",
        ),
    ] = "auto",
    formula_enable: Annotated[
        bool,
        Form(description="Enable formula parsing."),
    ] = True,
    table_enable: Annotated[
        bool,
        Form(description="Enable table parsing."),
    ] = True,
    image_analysis: Annotated[
        bool,
        Form(
            description=(
                "Enable image/chart analysis for VLM and hybrid backends. "
                "Hybrid medium effort automatically disables image/chart analysis."
            ),
        ),
    ] = True,
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
        Form(description="Return markdown content in response"),
    ] = True,
    return_middle_json: Annotated[
        bool,
        Form(description="Return middle JSON in response"),
    ] = False,
    return_model_output: Annotated[
        bool,
        Form(description="Return model output JSON in response"),
    ] = False,
    return_content_list: Annotated[
        bool,
        Form(description="Return content list JSON in response"),
    ] = False,
    return_images: Annotated[
        bool,
        Form(description="Return extracted images in response"),
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
    client_side_output_generation: Annotated[
        bool,
        Form(
            description=(
                "Defer final markdown/content-list generation to the client. "
                "When enabled, the server returns staged middle JSON, model output, and images."
            ),
        ),
    ] = False,
    start_page_id: Annotated[
        int,
        Form(description="The starting page for PDF parsing, beginning from 0"),
    ] = 0,
    end_page_id: Annotated[
        int,
        Form(description="The ending page for PDF parsing, beginning from 0"),
    ] = 99999,
) -> ParseRequestOptions:
    """解析 API/Router 共用的 multipart 表单，并保持 Swagger 参数同源。"""
    backend = validate_parse_backend(backend)
    effort = validate_parse_effort(effort)
    validate_public_http_client_request(
        public_bind_exposed=bool(
            getattr(request.app.state, "public_bind_exposed", False)
        ),
        allow_public_http_client=bool(
            getattr(request.app.state, "allow_public_http_client", False)
        ),
        backend=backend,
        server_url=server_url,
    )
    validate_parse_model_credentials(backend, model, api_key)
    if client_side_output_generation:
        return_md = False
        return_middle_json = True
        return_model_output = True
        return_content_list = False
        return_images = True

    effective_return_original_file = return_original_file and response_format_zip
    return ParseRequestOptions(
        files=files,
        lang_list=validate_parse_lang_list(lang_list),
        backend=backend,
        effort=effort,
        parse_method=validate_parse_method(parse_method),
        formula_enable=formula_enable,
        table_enable=table_enable,
        image_analysis=image_analysis,
        server_url=server_url,
        model=model,
        api_key=api_key,
        return_md=return_md,
        return_middle_json=return_middle_json,
        return_model_output=return_model_output,
        return_content_list=return_content_list,
        return_images=return_images,
        response_format_zip=response_format_zip,
        return_original_file=effective_return_original_file,
        client_side_output_generation=client_side_output_generation,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
    )
