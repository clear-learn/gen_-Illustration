# 베이스 이미지(EasyOCR를 수행할 때 GPU 사용)
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# 환경변수 설정
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 작업 디렉토리 설정
WORKDIR /workspace

# 필수 패키지 업데이트 및 설치
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    python3-pip \
    tzdata \
    unzip \
    wget \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    libxss1 \
    libappindicator1 \
    libindicator7 \
    libgl1-mesa-glx \
    xvfb \
    # cleanup
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# 크롬 브라우저 설치
RUN wget -q -O chrome-linux64.zip https://storage.googleapis.com/chrome-for-testing-public/127.0.6533.99/linux64/chrome-linux64.zip && \
    unzip chrome-linux64.zip && \
    mv chrome-linux64 /usr/local/chrome && \
    ln -s /usr/local/chrome/chrome /usr/local/bin/google-chrome && \
    chmod +x /usr/local/bin/google-chrome && \
    rm chrome-linux64.zip

# 크롬 드라이버 설치
RUN wget -q -O chromedriver-linux64.zip https://storage.googleapis.com/chrome-for-testing-public/127.0.6533.99/linux64/chromedriver-linux64.zip && \
    unzip chromedriver-linux64.zip && \
    mv chromedriver-linux64/chromedriver /usr/local/bin && \
    chmod +x /usr/local/bin/chromedriver && \
    rm chromedriver-linux64.zip

# pip 업그레이드
RUN python -m pip install --upgrade pip

# 소스 코드 복사
COPY . /workspace

# requirements.txt 설치
RUN python -m pip install -r /workspace/requirements.txt

CMD ["bash"]