#!/bin/bash
mkdir -p /mnt/hdd2/datasets_text/Open-Orca
mkdir -p /mnt/hdd2/datasets_text/PleIAs
mkdir -p /mnt/hdd2/datasets_text/Platypus

docker stop data_io_hrm_text_container 2>/dev/null
docker rm -f data_io_hrm_text_container 2>/dev/null
docker build -t data_io_hrm_text_image -f docker/DockerFile .

docker run -d --name data_io_hrm_text_container \
  --restart unless-stopped \
  --user $(id -u):$(id -g) \
  -v /mnt/hdd2/datasets_text:/mnt/hdd2/datasets_text \
  -w /mnt/hdd2/datasets_text \
  -e HF_TOKEN=$HF_TOKEN \
  -e HF_XET_HIGH_PERFORMANCE=1 \
  -e HF_HOME=/mnt/hdd2/datasets_text/.hf_cache \
  -e PYTHONUNBUFFERED=1 \
  data_io_hrm_text_image \
  bash -c "find /mnt/hdd2/datasets_text/ -name '*.lock' -delete 2>/dev/null; \
           \
           echo 'Starting FLAN download...'; \
           hf download Open-Orca/FLAN --repo-type dataset --local-dir /mnt/hdd2/datasets_text/Open-Orca/FLAN --max-workers 8; \
           \
           echo 'Starting SYNTH download...'; \
           hf download PleIAs/SYNTH --repo-type dataset --local-dir /mnt/hdd2/datasets_text/PleIAs/SYNTH --max-workers 8; \
           \
           echo 'Starting ARB download...'; \
           hf download imone/ARB --repo-type dataset --local-dir /mnt/hdd2/datasets_text/Platypus/ARB --max-workers 8; \
           \
           echo 'Cloning scibench...'; \
           git clone https://github.com/mandyyyyii/scibench.git /mnt/hdd2/datasets_text/Platypus/scibench || true; \
           \
           echo 'Downloading and extracting mathematics_dataset...'; \
           wget -O /mnt/hdd2/datasets_text/mathematics_dataset-v1.0.tar.gz 'https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz' && tar -xzvf /mnt/hdd2/datasets_text/mathematics_dataset-v1.0.tar.gz -C /mnt/hdd2/datasets_text && rm /mnt/hdd2/datasets_text/mathematics_dataset-v1.0.tar.gz; \
           \
           echo 'All downloads complete!'; \
           sleep infinity"