FROM nvidia/cuda:10.2-devel-ubuntu18.04
WORKDIR /root/blue-porcelain
COPY . .
ENV CUDA_INSTALL_PATH=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH CUDA_VERSION_NUMBER="10020" CUDA_VERSION_STRING="10.2" LD_LIBRARY_PATH="/root/blue-porcelain/runtime:/root/blue-porcelain/driver/csim:/"
RUN echo "==> Install make gcc ..." && \
sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
rm /etc/apt/sources.list.d/cuda.list && \
apt-get clean && \
apt-get update && \
apt-get install -y make gcc python3 bc numdiff && \
ln -s /usr/bin/python3 /usr/bin/python && \
echo "==> Install done"
