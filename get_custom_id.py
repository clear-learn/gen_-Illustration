import json
import os
import re
import requests

from dotenv import load_dotenv
from typing import Optional


#==============================================================
# 미드저니가 생성한 이미지의 고유키(custom_id)를 추출하는 클래스 : START
#==============================================================
class GetCustomId:
    def __init__(self, passed_prompt: str=None):
        self.passed_prompt = passed_prompt

        self._environment_initialization()
        self.custom_id = self.get_custom_id_corresponding_to_prompt()


    def _environment_initialization(self):
        '''
        ".env" 파일에서 환경변수를 로드하는 메서드
        '''
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(current_dir, ".env")

        # .env 파일 유효성 체크
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False, verbose=False)
        else:
            raise FileNotFoundError(f"환경변수를 '{env_path}'에서 로드할 수 없습니다.")

        ''' 환경변수 로드 '''
        # 디스코드 채팅 채널ID
        self.channel_id = os.getenv("CHANNEL_ID")
        # 디스코드 계정 TOKEN
        self.authorization = os.getenv("AUTHORIZATION")
        self.headers = {
            "authorization": self.authorization
        }


    def _extract_content_prompt(self, message: dict) -> Optional[str]:
        '''
        이미지 생성에 사용된 순수 프롬프트(pre-suffix 제외)를 추출하는 메서드

        Parameters:
        - message: 디스코드 메세지

        Returns:
        - str: 이미지 생성에 사옹된 순수 프롬프트문
        '''
        # JSON으로 parsing된 디스코드 메세지에서, 프롬프트문이 포함된 부분의 key는 "content"
        content = message.get("content", "")

        # 미드저니에서 이미지 생성에 사용한 프롬프트문은 **로 씌워져 있음
        prompt_pattern = re.compile(r"\*\*(.*?)\*\*")

        content_match = prompt_pattern.search(content)
        if content_match:
            extracted_content = content_match.group(1)

            # suffix를 제외한 순수 프롬프트문을 획득
            content_prompt = extracted_content.split("--")[0].strip()
            return content_prompt
        else:
            return None


    def _extract_custom_id(self, message: dict) -> Optional[str]:
        '''
        이미지가 생성되면 고유 식별키(custom_id)가 생성되는데, 이를 추출하는 메서드

        Parameters:
        - message: 디스코드 메세지

        Returns:
        - str : 고유 식별키(custom_id)
        '''
        # CustomId 정규식 (36자리의 16진수 및 하이픈 추출)
        custom_id_pattern = re.compile(r"[0-9a-zA-Z-]{36}")
        
        for dict_keys in message.get("components", []):
            for second_dict_keys in dict_keys.get("components", []):
                if "custom_id" in second_dict_keys:
                    match = custom_id_pattern.search(second_dict_keys["custom_id"])
                    if match:
                        return match.group(0)
        return None


    def get_custom_id_corresponding_to_prompt(self) -> Optional[str]:
        '''
        전달된 프롬프트(passed_prompt)에 대응되는 "custom_id"를 식별하는 메서드

        Returns:
        - str: 이미지 생성에 사용된 프롬프트문에 대응되는 "custom_id"
        '''
        def _clean_and_lowercase(text):
            # 알파벳을 제외한 모든 문자 제거
            cleaned_text = re.sub("[^a-zA-Z]", "", text)
            # 소문자로 변환
            lowercase_text = cleaned_text.lower()
            return lowercase_text

        try:
            # 참고: "limit=100"이 최대임, 최근 메세지부터 거슬러 올라가며 가져옴
            r = requests.get(f"https://discord.com/api/v10/channels/{self.channel_id}/messages?limit=10", headers=self.headers)

            # 요청이 성공했는지 체크
            r.raise_for_status()

            # JSON 파싱
            try:
                json_messages = r.json()
                # with open("temp_messages.json", "w", encoding="utf-8") as f:
                #     json.dump(json_messages, f, ensure_ascii=False, indent=4)
            except json.JSONDecodeError as e:
                print(f"디스코드 메세지 JSON 디코딩 에러")
                raise e

            for message in json_messages:
                content_prompt = self._extract_content_prompt(message=message)

                ''' 이게 문제였네 '''
                clean_content_prompt = _clean_and_lowercase(content_prompt)
                clean_passed_prompt = _clean_and_lowercase(self.passed_prompt)

                # print(f"content_prompt: {clean_content_prompt}")
                # print(f"passed_prompt: {clean_passed_prompt}")

                # 미드저니에 던진 프롬프트와 일치하는 메세지를 찾는다
                if clean_content_prompt and clean_content_prompt == clean_passed_prompt:
                    # 해당 메세지의 custom_id 추출
                    custom_id = self._extract_custom_id(message=message)

                    if custom_id:
                        return custom_id
        except requests.exceptions.RequestException as e:
            print(f"디스코드 HTTP 요청 실패: {e}")
            raise e

        return None
#==============================================================
# 미드저니가 생성한 이미지의 고유키(custom_id)를 추출하는 클래스 : END
#==============================================================