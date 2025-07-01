FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | sh

ENV PATH="/root/.pixi/bin:${PATH}"

RUN chmod -R 777 /root

COPY pixi.toml pixi.lock ./

RUN pixi install --locked -e extract

RUN pixi shell-hook --shell bash -e extract > /etc/profile.d/pixi.sh

CMD ["/bin/bash", "-c", ". /etc/profile.d/pixi.sh && exec /bin/bash"]