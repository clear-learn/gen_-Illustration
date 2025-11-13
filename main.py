import asyncio
import logging
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from typing import Optional

from prompt_sender import PromptSender
from get_custom_id import GetCustomId
from image_downloader import ImageDownloader


# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])

logger = logging.getLogger(__name__)


app = FastAPI()

class TestInput(BaseModel):
    chapter_summary: str
    coverimage_url: Optional[HttpUrl] = None

class ResultModel(BaseModel):
    custom_id: Optional[str]
    generated_prompt: Optional[str]
    image_url: Optional[str]
    local_filepath: Optional[str]
    # total_input_tokens: Optional[int]
    # total_output_tokens: Optional[int]

# main 함수 정의
async def generate_coverimage(test_input: TestInput) -> dict:

    chapter_summary = test_input.chapter_summary
    coverimage_url = test_input.coverimage_url

    prompt_sender = PromptSender(chapter_summary=chapter_summary, coverimage_url=coverimage_url)
    generated_prompt = prompt_sender.generated_prompt
    # total_input_tokens = prompt_sender.total_input_tokens
    # total_output_tokens = prompt_sender.total_output_tokens

    prompt_sender.send_prompt_to_midjourney()

    # 1차로 30초 대기(미드저니 fast 모드를 이용할 때 이미지 생성에 약 30초 소요)
    await asyncio.sleep(30)

    # 혹시 시간이 더 걸릴 수도 있으니, 10초씩 최대 20번 기다림 => 총 230초
    custom_id = None
    for _ in range(20):
        # 대기시간을 5초로 하면, "max retries exceeded with url" 에러 발생함
        await asyncio.sleep(10)
        get_custom_id = GetCustomId(passed_prompt=generated_prompt)
        custom_id = get_custom_id.custom_id

        # 정상적으로 custom_id 획득한 경우에 break
        if custom_id is not None:
            break

    if custom_id is None:
        raise Exception("생성된 이미지의 url을 확인할 수 없습니다. 처음부터 재시도해주세요.")

    # 정상적인 시나리오
    else:
        image_downloader = ImageDownloader(get_custom_id)

        image_url = image_downloader.image_url
        local_filepath = image_downloader.local_filepath

        # result = {
        #     "custom_id": custom_id,
        #     "generated_prompt": generated_prompt,
        #     "image_url": image_url,
        #     "local_filepath": local_filepath,
        #     "total_input_tokens": total_input_tokens,
        #     "total_output_tokens": total_output_tokens
        # }

        result = {
            "custom_id": custom_id,
            "generated_prompt": generated_prompt,
            "image_url": image_url,
            "local_filepath": local_filepath
        }
        return result


@app.post("/", response_model=ResultModel)
async def generate(test_input: TestInput):
    logger.info(f"fastapi 호출: {test_input}")
    result = await generate_coverimage(test_input)
    return ResultModel(**result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)