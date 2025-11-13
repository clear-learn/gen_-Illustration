# 책 삽화 및 일러스트레이션 생성

## 개요
ㅇ 챕터별 요약문과 기존 cover500 이미지를 이용하여, 챕터북의 표지를 생성
- 챕터 요약문 -> LLM(GPT)을 이용한 프롬프트문 생성 -> 미드저니를 통한 이미지 생성

ㅇ 현재 fastapi까지만 개발되어 있음(8000번 포트 사용)

<u>**ㅇ 환경변수 파일(".env")은, 요청시 별도로 전달하겠음**</u>

---
## 전반적인 Procedure

**1. prompt_sender.py**
- 챕터별 요약문 기반으로 이미지를 생성하기 위한 프롬프트문을 만들고, 이를 미드저니에 자동으로 전송

**2. get_custom_id.py**
- 미드저니에서 이미지가 생성된 후(fast모드에서 약 30초 소요), 고유 식별키(custom_id)를 추출

**3. image_downloader.py**
- custom_id를 이용하여, 미드저니를 통해 생성된 이미지url에 접근
  - 과거에는 selenium과 mitmproxy를 이용하여, 이미지를 중간에 가로채어 이미지 다운을 수행(cloudflare 보안 이슈 때문)
  - 하지만 현재는 이미지url에 접근하여, 이미지 다운이 아닌 1024*1024 사이즈의 screenshot을 찍는 것으로 변경
- ~~인물의 이상한 얼굴 묘사,~~ 이상한 글자(텍스트)가 있는지 체크
  - OpenCV 기반의 face detection 성능이 좋지 않아, 메서드는 코드에 있으나 적용하지는 않음
- 정상적인 이미지는 로컬에 저장

---
## 코드 실행방법
**1. 레포지토리 clone**

**2. Docker 이미지 빌드**

```bash
docker build -t {IMAGE_NAME} .
```

**3. Docker 컨테이너 실행**
- 기존에 다른 쪽에서 8000번 포트를 사용하고 있다면, main.py에서의 fastapi 포트 및 컨테이너 띄울 때의 포트 변경 필요함
```bash
docker run -it -p 8000:8000 --gpus all --name {CONTAINER_NAME} {IMAGE_NAME}:{TAG}
```

**4. 컨테이너 내에서 코드 실행**
```bash
python main.py
```

ㅇ 샘플 결과
- input : 챕터요약문(required), 표지이미지(optional)
- 표지이미지는 cover500 사용을 권장함

```bash
{
  "chapter_summary": "string",
  "coverimage_url": "https://example.com/"
}
{
  "chapter_summary": "소피에게 보내는 편지였다. 나는 마을을 떠났고, 시초지로 향하고 있었다. ......... 이제 내가 먼저 떠나는 이유를 이해해주길 바란다. 언젠가 지구에서 만나자.",
  "coverimage_url": "https://image.aladin.co.kr/product/25180/58/cover500/e082537439_1.jpg"
}
```


- output : 고유키, 이미지 생성에 사용된 프롬프트, 챕터북 표지이미지url, 해당 이미지의 로컬 경로(컨테이너상)
```bash
{
  "custom_id": "string",
  "generated_prompt": "string",
  "image_url": "string",
  "local_filepath": "string"
}
{
  "custom_id": "77ea8dd5-0270-4ca0-b755-3873e57da64a",
  "generated_prompt": "A mystical starting point of a pilgrimage in a remote, possibly sacred area surrounded by an aura of secrecy and tradition, featuring obscured pathways, ancient symbols or markers, and natural barriers like thick forests or mountains, in the style of Symbolism",
  "image_url": "https://cdn.midjourney.com/77ea8dd5-0270-4ca0-b755-3873e57da64a/0_0.png",
  "local_filepath": "...../images/77ea8dd5-0270-4ca0-b755-3873e57da64a_0.png"
}
```
---
## 작업 히스토리
**[2024-08-08]**

ㅇ v2_commit

ㅇ 주요 변경사항
- Dockerfile 수정(stable 버전의 크롬 브라우저 및 크롬 드라이버 직접 설치)
- 표지이미지를 사용하는 경우와 그렇지 않은 경우의 json payload 정의 분리
- 미드저니가 생성한 이미지의 추출 방식 변경
  - 기존: mitmproxy를 이용해 cdn 경로에서 직접 다운
  - 수정: cdn 경로에서 1024*1024 사이즈의 screenshot 수행

**[2024-07-23]**

ㅇ fastapi_v1 commit
- 환경설정 파일 및 Dockerfile 1차 업로드
