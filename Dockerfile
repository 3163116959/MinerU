# syntax=docker/dockerfile:1

# CPU-only 基础镜像。不用 vllm/vllm-openai：
# engine_utils._select_linux_engine() 只要 import vllm 成功就会选 vllm，
# 而无 CUDA 设备时 vllm 初始化必失败 → 保持 vllm 未安装，自动回落 transformers。
ARG BASE_IMAGE=python:3.12-slim-bookworm
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn

# ========== deps：外部依赖 → /opt/venv，只由依赖清单决定 ==========
FROM ${BASE_IMAGE} AS deps
ARG PIP_INDEX
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_DEFAULT_INDEX=${PIP_INDEX} \
    PIP_INDEX_URL=${PIP_INDEX} \
    PIP_NO_CACHE_DIR=1

RUN pip install uv

WORKDIR /src
# 只复制依赖解析所需文件 → 改业务代码不动这层。
# version.py / README.md / LICENSE.md 是 setuptools 生成项目元数据的输入，缺一则解析失败。
COPY pyproject.toml uv.lock README.md LICENSE.md ./
COPY mineru/version.py ./mineru/version.py

# 不加 --locked：uv.lock 记录的是 pypi.org URL，指向国内镜像源必然触发重解析，
# 二者互斥 → 这里选镜像源（构建速度优先）。要严格复现就改用 --locked 并去掉 UV_DEFAULT_INDEX。
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project --extra core

# ========== weights：烤入官方权重，与业务代码无依赖关系 ==========
# vlm      → MinerU2.5-Pro-2605-1.2B，高保真解析
# pipeline → DocLayout/OCR/公式/表格，供 pipeline 与 hybrid 模式使用
# 挂在 deps 上而非 build 上：只复制下载器的 import 闭包（7 个文件），
# 改业务代码这层结构上就不会失效。上游给这些模块加新 import → 这里直接 ImportError，
# 不静默降级。cache mount 只兜住依赖清单变动这类少数场景。
# 缓存路径与运行时路径同为 /opt/models → mineru.json 记录的绝对路径无需改写。
FROM deps AS weights
COPY mineru/__init__.py ./mineru/
COPY mineru/utils/__init__.py \
     mineru/utils/enum_class.py \
     mineru/utils/models_download_utils.py \
     mineru/utils/config_reader.py \
     ./mineru/utils/
COPY mineru/cli/__init__.py mineru/cli/models_download.py ./mineru/cli/
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/src \
    MODELSCOPE_CACHE=/opt/models \
    MINERU_TOOLS_CONFIG_JSON=/opt/models/mineru.json
# rm mineru.json：MinerU 的"配置里有路径且目录存在就跳过下载"是目录级判断
# （models_download_utils.py:151），断点续传留下的半截文件会被当成下载完成。
# 删掉配置 → 每次都交给 modelscope 自己校验补齐。
# 再用 .incomplete 断言兜底：宁可构建失败，也不把残缺权重烤进镜像。
RUN --mount=type=cache,target=/opt/models,id=mineru-models,sharing=locked \
    rm -f /opt/models/mineru.json \
    && python -m mineru.cli.models_download -s modelscope -m all \
    && ! find /opt/models -name '*.incomplete' | grep -q . \
    && cp -a /opt/models /opt/models-export

# ========== build：项目打成 wheel，装到 venv 之外的 /opt/app ==========
FROM deps AS build
COPY . /src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel --out-dir /src/dist \
    && uv pip install --python /opt/venv --no-deps --target /opt/app /src/dist/*.whl

# ========== runtime ==========
FROM ${BASE_IMAGE}
ARG APT_MIRROR

# apt 换国内源：只替换主机名，仓库路径（/debian、/debian-security）原样保留。
# 用 http：apt 靠 GPG 验签，不依赖 TLS，省掉 ca-certificates 的坑。
# 海外构建传 --build-arg APT_MIRROR= 留空即跳过。
RUN if [ -n "${APT_MIRROR}" ]; then \
        for f in /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources; do \
            if [ -f "$f" ]; then \
                sed -i "s|https\?://deb.debian.org|http://${APT_MIRROR}|g; \
                        s|https\?://security.debian.org|http://${APT_MIRROR}|g" "$f"; \
            fi; \
        done; \
    fi

# 系统依赖：opencv 需要 libgl1 + libglib2.0-0，CJK 渲染需要 noto 字体
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1 \
        libglib2.0-0 && \
    fc-cache -f && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 三条 COPY = 三层，按变动频率升序：权重（几乎不变）→ 依赖 → 业务代码。
# 依赖层取自 deps（不含项目代码），改代码时 digest 不变，pull 只重传几 MB 的 /opt/app。
COPY --from=weights /opt/models-export /opt/models
COPY --from=deps /opt/venv /opt/venv
COPY --from=build /opt/app /opt/app

ENV PATH="/opt/venv/bin:/opt/app/bin:$PATH" \
    PYTHONPATH=/opt/app \
    PYTHONUNBUFFERED=1 \
    MINERU_TOOLS_CONFIG_JSON=/opt/models/mineru.json \
    MINERU_MODEL_SOURCE=local \
    MINERU_DEVICE_MODE=cpu

WORKDIR /app
EXPOSE 8000

ENTRYPOINT ["/bin/bash", "-c", "exec \"$@\"", "--"]
CMD ["mineru-api", "--host", "0.0.0.0", "--port", "8000"]
