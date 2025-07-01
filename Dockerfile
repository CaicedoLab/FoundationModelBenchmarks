FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

WORKDIR /Benchmarking

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

ENV PIXI_HOME=/home/appuser/.pixi

RUN mkdir -p ${PIXI_HOME} && chown -R appuser:appuser /home/appuser
ENV PATH="${PIXI_HOME}/bin:${PATH}"
RUN curl -fsSL https://pixi.sh/install.sh | sh

COPY --chown=appuser:appuser pixi.toml pixi.lock ./

USER appuser

RUN pixi install --locked -e extract

USER appuser

CMD ["/bin/bash", "-c", ". /etc/profile.d/pixi.sh && exec /bin/bash"]