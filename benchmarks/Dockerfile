FROM rockylinux:9

# Set working directory
WORKDIR /app

# Install dependencies
RUN dnf update -y && \
    dnf install -y epel-release && \
    dnf install -y \
    java-21-openjdk-headless \
    wget \
    git \
    python3 \
    python3-pip \
    && dnf clean all

# Verify Java version
RUN java -version

# Download SPMF and install Python library
RUN wget http://www.philippe-fournier-viger.com/spmf/SPMF.jar -O spmf.jar && \
    git clone https://github.com/danxuZhang/parallelGSP.git && \
    cd parallelGSP && \
    pip install .

# Create necessary directories
RUN mkdir -p /app/data /app/output

# Copy your files
COPY FIFA.txt /app/data/
COPY benchmarks.sh /app/
COPY pgsp.py /app/

CMD ["bash"]

