FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y curl tar git && rm -rf /var/lib/apt/lists/*

# Pull Mora binary from official release
RUN curl -L -o /tmp/mora-docker.tar.gz \
    https://github.com/Microbiosis/mora-lang/releases/download/v0.111.0/mora-docker-amd64.tar.gz \
    && mkdir -p /tmp/mora-extract /tmp/mora-rootfs \
    && tar -xzf /tmp/mora-docker.tar.gz -C /tmp/mora-extract \
    && cd /tmp/mora-extract \
    && for layer in blobs/sha256/871d48e9bf6b789a439a4a19c1e152009924d818e60f7c51e498573223cd7cda \
                    blobs/sha256/be9c7b776e732ca043049275ec7c8d1c27797fd37d482bbb7655006fbce6a876 \
                    blobs/sha256/8bae65e009a9e73e09cd7e9c2439994c76b26c907a6d84a9e112d532e1022021; do \
         tar -xf "$layer" -C /tmp/mora-rootfs 2>/dev/null || true; \
       done \
    && cp /tmp/mora-rootfs/usr/local/bin/mora /usr/local/bin/mora \
    && chmod +x /usr/local/bin/mora \
    && rm -rf /tmp/mora-*

WORKDIR /work
RUN git clone --depth 1 https://github.com/Microbiosis/mora-llm-diffusion.git /work \
    && mkdir -p /work/checkpoints

CMD ["mora", "src/train.mora"]
