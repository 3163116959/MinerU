# syntax=docker/dockerfile:1

# CPU-only 基础镜像。不用 vllm/vllm-openai：
# engine_utils._select_linux_engine() 只要 import vllm 成功就会选 vllm，
# 而无 CUDA 设备时 vllm 初始化必失败 → 保持 vllm 未安装，自动回落 transformers。
# 基础镜像用 trixie（Debian 13）而非 bookworm：LibreOffice 从 7.4.7 跳到 25.2.3，
# EMF/WMF 导入与渲染差了三年的修复，直接决定 rasterize_vector_image 的成图质量。
# 锁 digest：python:3.12-slim-trixie 上游每周重建，只锁 tag 会让同一份 Dockerfile
# 在 tag 刷新后全阶段缓存失效。升级基础镜像时手动换 digest。
ARG BASE_IMAGE=python:3.12-slim-trixie@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9
ARG PIP_INDEX=https://repo.huaweicloud.com/repository/pypi/simple
ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn

# ========== system：系统层，只由本段的 apt 清单决定 ==========
# 单独成段并作为 runtime 的父层：Python 依赖、权重、业务代码的任何变动
# 都在它之上，改 pyproject.toml 不会重建这 ~800MB 的系统层。
FROM ${BASE_IMAGE} AS system
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

# opencv 需要 libgl1 + libglib2.0-0，CJK 渲染需要 noto 字体；
# LibreOffice：WMF/EMF 矢量图在非 Windows 上 Pillow 无渲染后端，
# office_image.rasterize_vector_image 调 soffice 转 PNG，缺它则退占位图。
# ffmpeg：whisperx.load_audio 调 ffmpeg 解码，torchcodec 也要它的共享库。
# 一段 apt 装完：拆成两段会各跑一次 apt-get update，且前一段清 lists 会让后一段白跑索引。
# apt 缓存挂载需先删 docker-clean，否则 apt 装完即清空缓存目录；
# 挂载的 /var/cache/apt 与 /var/lib/apt/lists 不落进镜像层，无需再手工清理。
# 时区 UTC+8、编码 UTF-8 一并在此固化（C.UTF-8 是 glibc 内建，无需 locales 包）。
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libreoffice-draw \
        tzdata && \
    fc-cache -f && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone

# ========== downloader：只含模型下载器源码，供 models 阶段 bind 挂载 ==========
# 单独成段：models 层的缓存键只取决于这几个文件，改其它业务代码不会触发重下权重。
# 上游给这些模块加新 import → models 阶段直接 ImportError，不静默降级。
FROM scratch AS downloader
COPY mineru/__init__.py /src/mineru/
COPY mineru/utils/__init__.py \
     mineru/utils/enum_class.py \
     mineru/utils/models_download_utils.py \
     mineru/utils/config_reader.py \
     /src/mineru/utils/
COPY mineru/cli/__init__.py mineru/cli/models_download.py /src/mineru/cli/

# ========== models：权重直接下载进镜像层，紧贴系统层之上 ==========
# 权重几乎不变 → 放在依赖与业务代码之下，改 pyproject.toml / 代码都复用这层，
# 推送/拉取时也只在系统层或下载器变化时才重传这几 GB。
# 下载器依赖装进临时 --target 目录，同一 RUN 内删掉，不污染运行环境；
# HOME 指向临时目录 → modelscope/nltk 等写到家目录的杂项随之一并清掉。
# 末尾 .incomplete 断言：宁可构建失败，也不把残缺权重烤进镜像。
FROM system AS models
ARG PIP_INDEX
RUN --mount=type=bind,from=downloader,source=/src,target=/opt/downloader \
    --mount=type=cache,target=/var/cache/pip \
    set -eux; \
    PIP_CACHE_DIR=/var/cache/pip python -m pip install --index-url "${PIP_INDEX}" \
        --target /tmp/dl click loguru requests huggingface-hub modelscope nltk; \
    HOME=/tmp/dlhome \
    PYTHONPATH=/tmp/dl:/opt/downloader \
    MODELSCOPE_CACHE=/opt/models \
    MINERU_AUDIO_MODELS_DIR=/opt/models/audio \
    MINERU_TOOLS_CONFIG_JSON=/opt/models/mineru.json \
        python -m mineru.cli.models_download -s modelscope -m all; \
    ! find /opt/models -name '*.incomplete' | grep -q .; \
    rm -rf /tmp/dl /tmp/dlhome

# ========== deps：外部依赖 → /opt/venv，只由依赖清单决定 ==========
# 直接挂在基础镜像上：装 wheel 不需要系统层那些运行时库，
# 与 system 分叉可让两边并行构建。
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

# ========== build：项目打成 wheel，装到 venv 之外的 /opt/app ==========
FROM deps AS build
COPY . /src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel --out-dir /src/dist \
    && uv pip install --python /opt/venv --no-deps --target /opt/app /src/dist/*.whl

# ========== runtime ==========
# 从 models 起（系统层 → 权重层），这一阶段自身不再有 RUN 层，只剩两条 COPY。
FROM models

# 两条 COPY = 两层，按变动频率升序：依赖 → 业务代码。
# 依赖层取自 deps（不含项目代码），改代码时 digest 不变，pull 只重传几 MB 的 /opt/app。
COPY --from=deps /opt/venv /opt/venv
COPY --from=build /opt/app /opt/app

ENV PATH="/opt/venv/bin:/opt/app/bin:$PATH" \
    PYTHONPATH=/opt/app \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    MINERU_TOOLS_CONFIG_JSON=/opt/models/mineru.json \
    NLTK_DATA=/opt/models/audio/nltk_data \
    PYANNOTE_METRICS_ENABLED=false \
    MINERU_MODEL_SOURCE=local \
    MINERU_DEVICE_MODE=cpu

WORKDIR /app
EXPOSE 8000

ENTRYPOINT ["/bin/bash", "-c", "exec \"$@\"", "--"]
CMD ["mineru-api", "--host", "0.0.0.0", "--port", "8000"]
