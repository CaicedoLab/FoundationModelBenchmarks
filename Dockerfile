FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | sh

ENV PATH="/root/.pixi/bin:${PATH}"

RUN chmod -R 777 /root

COPY pixi.toml pixi.lock ./

RUN CONDA_OVERRIDE_CUDA="12.4" pixi install --locked -e extract

RUN pixi shell-hook --shell bash -e extract > /etc/profile.d/pixi.sh

CMD ["/bin/bash", "-c", ". /etc/profile.d/pixi.sh && exec /bin/bash"]