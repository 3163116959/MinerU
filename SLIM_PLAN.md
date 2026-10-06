# MinerU + kb-demo 精简计划

> 单一事实来源。两个项目的改动都按本文推进，每完成一步更新「进度」勾选与「执行记录」。
> kb-demo 侧指针：`/Users/apple/PythonProject/kb-demo/docs/mineru-slim.md`

## 0. 目标与边界

- 目标：MinerU 只服务 kb-demo 当前用法 → 接口参数收敛、删除不用的后端/入口/依赖。
- 硬约束：kb-demo 现有功能行为不变（解析结果、切分、入库一致）。
- 原则：
  - 只删确定不需要的；需要的不动；**不确定的暂不动**（列入第 5 节）。
  - 不重构内部逻辑，只删分支/入口/参数。
  - 不考虑合并上游。
  - 两项目同步改，SDK 字段与服务端表单字段保持一致。

## 1. 现状基线：kb-demo 实际调用契约

入口 `POST /knowledge-bases/{kb_id}/documents` → `document_service.upload` → 按后缀分流：

| 文件类型 | kb-demo 函数 | MinerU 接口 | 表单参数 |
|---|---|---|---|
| md/markdown | `save_markdown` | 不调用 | — |
| pdf / 图片 / office | `parse_to_markdown` | `POST /file_parse` | backend=hybrid-http-client, server_url, model, api_key, effort=medium, image_analysis=true, table_enable=true, formula_enable=true, return_md=true, return_middle_json=true, return_images=true, client_side_output_generation=false |
| 音视频 | `parse_audio` | `POST /file_parse` | return_md=true, return_middle_json=true（backend 走服务端默认，音频分支不看 backend） |

未传 → 服务端默认：parse_method=auto, lang_list=["ch"], start/end_page_id=全文。
使用的响应字段：`results.<name>.md_content / middle_json / images`。

部署：`docker-compose.yml` → `mineru-api --host 0.0.0.0 --port 8000 --enable-vlm-preload true --allow-public-http-client`，
CPU 模式，环境 `MINERU_VLM_GENERIC_IMAGE_PROMPT=1`。

已核实行为（hybrid_analyze.py:957-977）：medium 分支把 `image_analysis` 原样传给 VLM，不受 medium 限制 → 写死 image_analysis=true 与现状一致。

## 2. 必须保留（不动）

- `/file_parse`、`/health`；`fast_api.py` 主流程 `run_parse_job` → `aio_do_parse`。
- 输入：PDF、图片（转 PDF）、office（docx/pptx/xlsx/doc/ppt/xls/odt/ods/odp/rtf/epub/csv）、音视频（WhisperX）。
- hybrid 后端 medium 分支全部链路：
  - `backend/hybrid/*`（删 high 分支除外）
  - `backend/pipeline/*` 的模型与工具（hybrid 依赖 layout/OCR/MFD/MFR/表格/`txt_spans_extract`）
  - `backend/vlm/*` 中 http-client 推理与 middle_json/mkcontent 相关
  - `model/*` 中 layout/ocr/mfr/table/docx/pptx/xlsx
- office 图片分析：`_resolve_office_vlm_backend("hybrid-http-client")` → `http-client`。
- `pdf_classify`（auto 判 txt/ocr）、`parse_method` 内部三态逻辑（只删对外参数）。
- 内部 `lang` 透传（只删对外参数，固定 `["ch"]`）。
- 公网策略 `public_http_client_policy` + `--allow-public-http-client`（compose 依赖）。
- 本地已有未提交改动（audio_analyze / enum_class / models_download / Dockerfile 等）。

## 3. 删除清单（确定）

### 3.1 API 参数（`mineru/cli/api_request.py` + `fast_api.py`）

| 参数 | 处理 | 服务端固定值 |
|---|---|---|
| backend | 删表单字段 | `hybrid-http-client` |
| effort | 删 | `medium` |
| parse_method | 删 | `auto` |
| lang_list | 删 | `["ch"]` |
| formula_enable / table_enable / image_analysis | 删 | `true` |
| client_side_output_generation | 删 | `false` |
| return_model_output / return_content_list | 删 | `false` |
| start_page_id / end_page_id | 删 | 全文 |
| server_url / model / api_key | **保留** | 请求传入（现状） |
| return_md / return_middle_json / return_images | **保留** | 音频与文档请求取值不同 |

注意：音频请求不带 server_url。固定 backend 后，校验逻辑不得要求音频请求带 server_url（第 4 阶段验证点）。

### 3.2 后端分支
- `backend_options.py`：只留 `hybrid-http-client` 与 `medium`。
- `common.py`：`do_parse/aio_do_parse` 的 pipeline / vlm / hybrid-engine 派发分支（`_process_pipeline`、`_process_vlm`、`_async_process_vlm`、本地 engine 路径）。
- `hybrid_analyze.py`：`effort == "high"` 分支（`batch_two_step_extract` 调用处）。

### 3.3 入口与周边
- CLI 入口：`cli/client.py`（`mineru` 命令）、`cli/api_client.py`（仅被 client/gradio/router 引用）。
- `cli/gradio_app.py`、`cli/router.py`。
- 本地 VLM 服务：`cli/vlm_server.py`、`model/vlm/vllm_server.py`、`model/vlm/lmdeploy_server.py`。
- `pyproject.toml`：
  - scripts 删 `mineru`、`mineru-vllm-server`、`mineru-lmdeploy-server`、`mineru-openai-server`、`mineru-router`、`mineru-gradio`
  - 保留 `mineru-api`、`mineru-models-download`
  - extras 删 `vllm`、`lmdeploy`、`mlx`、`gradio`；`core` 去掉 `mineru[gradio]`
  - `all` 同步调整

## 4. 分阶段执行

每阶段结束：`python -c "import mineru.cli.fast_api"` 通过 → 起服务 → 跑第 6 节回归 → 与基线 diff 一致 → git commit。

- **阶段 0：基线**
  - MinerU：把现有未提交改动先 commit，新建分支 `slim`。
  - kb-demo：当前未纳入 git → `git add` 初次提交（排除 storage/output/pg 等大目录，先确认 .gitignore）。
  - 跑第 6 节回归，产物存 `kb-demo/tests/files/baseline/`。
- **阶段 1：删入口**
  - 删 3.3 中的文件与 scripts。
  - grep 确认 fast_api 无引用。
- **阶段 2：收敛 API 参数**
  - 改 3.1 的 `api_request.py`、`ParseRequestOptions`、`fast_api.py` 引用处。
  - `/tasks` 系列暂保留（见 5）。
- **阶段 3：删后端派发分支**
  - 改 3.2：common.py 派发、backend_options、hybrid high 分支。
  - 删完再 grep 每个被删函数的符号，保证无残留引用。
- **阶段 4：kb-demo 同步**
  - `src/sdk/models.py`：`ParseOptions` 只留 server_url/model/api_key/return_md/return_middle_json/return_images（+ /tasks、zip 若保留则留对应字段）；删 `Backend/Effort/ParseMethod/Lang` 枚举；`sdk/__init__.py` 同步导出。
  - `src/service/parse_service.py:build_options`：去掉 backend/effort/image_analysis/table/formula/client_side 字段；`server_url/model/api_key` 改为无条件传。
  - `src/config/settings.py`：删 `BACKEND`、`EFFORT`、`CLIENT_SIDE_OUTPUT_GENERATION`。
  - tests：`test_client.py` / `test_integration.py` 删 backend/effort 用法。
  - 验证音频请求不带 server_url 仍成功。
- **阶段 5：依赖与镜像**（先验证，再删）
  - pyproject extras 清理（3.3）→ `uv lock` → Docker 构建通过。
  - 第 5 节中验证通过的项再执行。

## 5. 不确定 → 暂不动（需验证后再决定）

| 项 | 疑点 | 验证方法 |
|---|---|---|
| `--enable-vlm-preload` + 本地 VLM 权重（Dockerfile 校验 `vlm` 目录） | hybrid-http-client 似乎不用本地 VLM，preload 可能纯浪费内存/冷启动 600s | 去掉 preload 启动 → 跑回归一致 → 再删权重下载与 `vlm_preload.py` |
| `vlm_analyze.py` 内本地 engine（transformers/vllm/lmdeploy/mlx）分支、`engine_utils.py` | 与 http-client 共用 `ModelSingleton` | 阶段 3 后看引用，单独一步删 |
| `/tasks`、`/tasks/{id}`、`/tasks/{id}/result` | kb-demo 生产不用，仅 integration 测试用 | 用户确认 |
| `response_format_zip` / `return_original_file` | 仅 kb-demo 测试 `parse_zip` 用 | 用户确认 |
| `pipeline_middle_json_mkcontent.py`、`llm_aided.py`、`visualization.py`、`draw_bbox.py`、`client_side_output.py`、`backend/anydoc` | 被 common.py / pipeline 内部引用，关系未理清 | 阶段 3 后逐个 import 检查 |
| `docs/`、`demo/`、`projects/`、`mkdocs.yml`、`docker/` | 非运行代码 | 用户确认 |
| 公网 http-client 策略 | compose 已显式放开；若改为服务端配置 VLM 凭证可删 | 另议 |

## 6. 回归用例（每阶段跑，结果对比基线）

样本放 `kb-demo/tests/files/`：

1. 纯文本 PDF
2. 图文混合 PDF
3. 扫描件 PDF
4. 图片（png/jpg）
5. docx（含图）
6. xlsx / pptx 各一
7. 音频（mp3）
8. md

对比项：`md_content` 文本 diff、图片数量、`middle_json` 块数、`_backend`/`_effort`/`_ocr_enable`。
VLM 输出有随机性 → 文本若有细微差异，以块数/结构为准，并人工抽查。

## 7. 风险

- 删 common.py 派发时误删 hybrid 共用函数 → 每删一个符号先 grep 引用。
- 固定 backend 后音频请求被 http-client 校验或公网策略拦截 → 阶段 2/4 专测。
- uv lock 重解析导致依赖版本漂移 → 阶段 5 单独 commit，便于回滚。
- kb-demo 无 git 历史 → 阶段 0 必须先提交。

## 进度

- [x] 阶段 0 基线
- [x] 阶段 1 删入口（含于阶段 3 回归）
- [x] 阶段 2 收敛 API 参数（含于阶段 3 回归）
- [x] 阶段 3 删后端派发分支
- [x] 阶段 4 kb-demo 同步
- [ ] 阶段 5 依赖与镜像
- [ ] 第 5 节逐项验证

## 执行记录

（每步完成后追加：日期、改动文件、验证结果、commit）

- 2026-10-06 阶段 0：MinerU 建分支 `slim`（基于 `d5a8b45c`）；kb-demo 初次提交 `4d75b34`；镜像打 tag `mineru-local:baseline`（`c91a2eaf346c`）；回归脚本 `kb-demo/scripts/slim_regress.py`，样本 `/tmp/slim_samples`。
  - 回归基线：audio 通过；demo1.pdf 起失败 → `server_url` 远程 VLM 不可达（宿主机连接超时，容器内 DNS 解析失败）。**需恢复网络后重跑基线**（用 baseline 镜像）。
- 2026-10-06 阶段 1：commit `49e7403b`，删 CLI/gradio/router/本地 VLM server 入口及 scripts。import 检查通过。
- 2026-10-06 阶段 2：commit `2b7575c3`。
  - `api_request.py`：表单只留 files/server_url/model/api_key/return_md/return_middle_json/return_images/response_format_zip/return_original_file；固定 `PARSE_BACKEND=hybrid-http-client`、`PARSE_METHOD=auto`、`PARSE_LANG=ch`；删 backend/effort/parse_method/lang 校验与 model 凭证校验。
  - `fast_api.py`：`AsyncParseTask` 去对应字段；`run_parse_job` 固定 effort=medium、formula/table/image=true、model_output/content_list=false、全页；去 pipeline→`do_parse` 分支；结果构建去 model_output/content_list；响应 `backend` 键保留（kb-demo 读取），值固定。
  - 验证：ruff F 通过、import 通过、OpenAPI 表单字段核对通过。端到端回归待网络恢复。
  - 行为变化：公网绑定且未开 `--allow-public-http-client` 时，音频请求也会被拒（backend 固定为 http-client）。compose 已开该开关 → 现部署无影响。
  - kb-demo 仍发旧字段 → FastAPI 忽略多余表单字段，兼容；阶段 4 再清理。
- 2026-10-06 阶段 0 补：网络恢复后用 baseline 镜像重跑，产物 `kb-demo/tests/files/baseline/`（8 样本全过；md 本地处理不经 MinerU）。
- 2026-10-06 阶段 3：commit `95df51a9`。
  - `common.py`：删 `do_parse`、`_process_pipeline`、`_process_vlm`、`_async_process_vlm`、`_process_hybrid`、`_resolve_office_vlm_backend`、`_prepare_pdf_bytes` 及 pipeline/vlm/engine 派发；`aio_do_parse` 只走 hybrid-http-client（音频 → office → hybrid），office 图片分析固定 `http-client`。
  - `hybrid_analyze.py`：删同步 `doc_analyze`、`_validate_parse_effort`、`HYBRID_ANALYZE_EFFORTS`、`effort == "high"` 分支（`aio_batch_two_step_extract`）；medium 逻辑去条件内联，middle_json `_effort` 仍由默认值写 `medium`。
  - 删 `backend_options.py`；`api_request.PARSE_BACKEND` 内联字面量。
  - 验证：被删符号 grep 无残留（`build/` 为旧构建产物，忽略）；ruff F 通过；新镜像内 import 通过；`docker compose up -d` healthy。
  - 回归（`tests/files/slim_p3/`）vs 基线：`report.json` 去耗时后完全一致（页数/块类型计数/图片数/`_backend`/`_effort`/`_ocr_enable`）；audio、xlsx md 字节一致；其余 md 差异仅在 VLM 图片描述与 OCR 空白（随机性），结构一致。
- 2026-10-06 阶段 4：kb-demo commit `6f15af5`（`slim_regress.py` 一并纳入）。
  - `sdk/models.py`：删 `Backend/Effort/ParseMethod/Lang`；`ParseOptions` 只留 server_url/model/api_key/return_md/return_middle_json/return_images/response_format_zip/return_original_file（与 MinerU 表单一致）；`to_form` 去 lang_list 分支。`sdk/__init__.py` 同步。
  - `parse_service.build_options`：无条件带 server_url/model/api_key（缺凭证仍显式报错）；去 backend/effort/image/table/formula/client_side。
  - `settings.py`：删 `BACKEND`、`EFFORT`、`CLIENT_SIDE_OUTPUT_GENERATION`。
  - tests：去 backend/effort 用法；pytest 24 passed / 8 skipped（集成测试需 `MINERU_TEST_URL`）。
  - 回归（`tests/files/slim_p4/`）vs 基线：`report.json` 去耗时后完全一致；音频请求不带 server_url 通过。
