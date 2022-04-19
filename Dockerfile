FROM nvidia/cuda:10.2-devel-ubuntu18.04
RUN echo "==> Install make gcc git ssh-server..." && \
sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
apt-get clean && \
apt-get update && \
apt-get install -y make gcc git wget zlib1g-dev openssh-server bc numdiff && \
cd ~ && \
ln -s /usr/bin/python3 /usr/bin/python && \
echo "export CUDA_INSTALL_PATH=/usr/local/cuda" >> /root/.bashrc && \
echo "export PATH=/usr/local/cuda/bin:$PATH" >> /root/.bashrc && \
echo "==> Install done"
