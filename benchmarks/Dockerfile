FROM rockylinux:9

WORKDIR /app

RUN dnf update -y && \
    dnf install -y \
    java-11-openjdk-headless \
    wget \
    git \
    python3 \
    python3-pip \
    && dnf clean all

RUN wget http://www.philippe-fournier-viger.com/spmf/SPMF.jar -O spmf.jar

RUN git clone https://github.com/danxuZhang/parallelGSP.git \
    && cd parallelGSP && pip install .

RUN mkdir -p /app/data /app/output

COPY FIFA.txt /app/data/

COPY benchmarks.py /app/data
COPY spmf_run.sh /app/data

CMD [bash]
