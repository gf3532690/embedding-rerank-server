FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖（torch 已由基础镜像提供，pip 安装时跳过）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY app/ ./app/
COPY config.yaml .

# 健康检查
HEALTHCHECK --interval=15s --timeout=10s --retries=5 --start-period=120s \
    CMD curl -f http://localhost:7997/health || exit 1

EXPOSE 7997 7998

ENTRYPOINT ["python", "-m", "app.main"]
