import json
import os
import re
import requests

from dotenv import load_dotenv
from openai import OpenAI
from typing import Any, Dict, Optional, Tuple



#======================================================================
# 챕터 요약문을 이용해 프롬프트문을 생성하고, 이를 미드저니에 전달하는 클래스 : START
#======================================================================
class PromptSender:
    def __init__(self, chapter_summary: str=None, coverimage_url: str=None):
        self.chapter_summary = chapter_summary
        self.coverimage_url = coverimage_url

        self._environment_initialization()
        self._config_initialization()
        self.generated_prompt, self.total_input_tokens, self.total_output_tokens = self.select_best_prompt_via_gpt(text=self.chapter_summary)
        # self.generated_prompt = self.select_best_prompt_via_claude()


    def _environment_initialization(self) -> None:
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
        # OpenAI api key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=self.openai_api_key)

        # # Claude api key
        # self.claude_api_key = os.getenv("CLAUDE_API_KEY")
        # self.claude_client = anthropic.Anthropic(api_key=self.claude_api_key)

        # 디스코드 채팅 채널ID
        self.channel_id = os.getenv("CHANNEL_ID")
        # 디스코드 계정 TOKEN
        self.authorization = os.getenv("AUTHORIZATION")
        # 디스코드 미드저니봇ID
        self.application_id = os.getenv("APPLICATION_ID")
        # 디스코드 채팅 서버ID
        self.guild_id = os.getenv("GUILD_ID")
        # 아래 3개 변수의 값은, 브라우저 '개발자도구'에서 확인
        self.session_id = os.getenv("SESSION_ID")
        self.version = os.getenv("VERSION")
        self.id = os.getenv("ID")
        # 미드저니 프롬프트 suffix
        self.suffix = os.getenv("SUFFIX")
        # 표지이미지 사용할 때 필요한 suffix
        self.image_suffix = os.getenv("IMAGE_SUFFIX")


    def _config_initialization(self) -> None:
        '''
        JSON 파일에서 configuration을 로드하는 메서드
        '''
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config_prompt_sender.json")

        # 파일 유효성 체크
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # GPT 모델
            self.gpt_model_name = config["gpt_model_name"]
            # 챕터 요약문에서 주요 요소 추출할 때의 GPT temperature
            self.feature_temperature: float = config["feature_temperature"]
            # 주요 요소를 이용하여 프롬프트문을 생성할 때의 GPT temperature
            self.prompt_temperature: float = config["prompt_temperature"]
            ''' 추가'''
            self.dry_temperature: float = config["dry_temperature"]
            # 여러 프롬프트문 가운데, 최적의 프롬프트문을 선택할 때의 GPT temperature
            self.common_temperature: float = config["common_temperature"]
            # 하나의 챕터 요약문을 이용해 생성할 프롬프트문의 개수
            self.number_of_prompts: int = config["number_of_prompts"]
            # (공통) LLM 호출에 실패하는 경우를 대비하기 위한 재시도 횟수
            self.number_of_retries: int = config["number_of_retries"]
        else:
            raise FileNotFoundError(f"설정을 '{config_path}'에서 로드할 수 없습니다.")


    def _extract_text_features_via_gpt(self, text: str) -> Optional[Tuple[Dict[str, Any], int, int]]:
        '''
        주어진 텍스트에서 이미지 생성을 위한 주요 요소를 추출하는 메서드

        Parameters:
        - text(str): 분석할 텍스트(챕터 요약문)

        Returns:
        - Optional[Tuple[Dict[str, Any], int, int]]: 중요한 장소, 핵심 객체, 제안하는 그림 스타일 등이 포함된 JSON 객체 / 입력 토큰 수 / 출력 토큰 수
        - 분석에 실패할 경우 None을 리턴
        '''
        #===== GPT 프롬프트 엔지니어링: START =====
        rules = """
        1. From the given text, select the most important location to be used as the background for an image.
        2. Utilize the content of the text as much as possible to describe the location with a focus on visual elements. Express the description in the form of noun phrases.
        3. Select key elements that can be included in the analyzed most important location and describe them.
        4. Exclude people(figures) as key elements.
        5. Considering the atmosphere of the most important location selected above, suggest a 'painting style'.
        """
        # 4. If the key element includes a person, infer the person's gender and age group based on the given text. If the gender or age group is unclear, exclude the person from the key elements.

        assistant_description = f"""You are an AI assistant specialized in analyzing text, mainly summaries of certain books. \
        Your task is to select the most important location and describe it along with key elements and suggesting a painting style. \
        For the analysis, keep the following rules in mind:

        Rules: {rules}
        """

        user_message  = f"""Select the most important location from the given text below and describe it along with key elements and suggesting a painting style in JSON format. \n{text}\n """
        #===== GPT 프롬프트 엔지니어링: END =====

        retry = 0
        step1_input_tokens = 0
        step1_output_tokens = 0

        while retry < self.number_of_retries:
            retry += 1
            try:
                response = self.openai_client.chat.completions.create(
                    model = self.gpt_model_name,
                    temperature = self.feature_temperature,
                    max_tokens = 1000,
                    messages = [
                        {
                            "role": "system",
                            "content": assistant_description
                        },
                        {
                            "role": "user",
                            "content": user_message
                        }
                    ],
                    response_format = {
                        "type": "json_object"
                    },
                    )

                ''' "finish_reason"에 따른 예외처리 '''
                # 최대 토큰 수를 초과한 경우
                if response.choices[0].finish_reason == "length":
                    print("토큰 개수가 초과되었습니다. 텍스트의 핵심 요소 재추출이 필요합니다.")
                    return None
                # OpenAI 정책에 의해 결과가 필터링된 경우
                elif response.choices[0].finish_reason == "content_filter":
                    print("생성된 결과가 OpenAI 정책에 따라 필터링되었습니다. 텍스트의 핵심 요소 재추출이 필요합니다.")
                    return None
                else:
                    analysis = response.choices[0].message.content
                    text_json_analysis = json.loads(analysis)

                    # input(prompt_tokens) / output(completion_tokens) 획득
                    step1_input_tokens += response.usage.prompt_tokens
                    step1_output_tokens += response.usage.completion_tokens

                    return text_json_analysis, step1_input_tokens, step1_output_tokens
            except Exception as e:
                print(f"retry={retry}: 요약문으로부터 주요 요소 추출을 재시도합니다. 에러: {e}")
                continue
        raise Exception("수차례 재시도했음에도 주요 요소 추출에 실패했습니다.")


    def _generate_normal_prompt_based_on_text_features_via_gpt(self, text_json_analysis: Dict[str, Any]) -> Optional[Tuple[str, int, int]]:
        '''
        챕터 요약문으로부터 분석된 핵심 요소(JSON format)를 이용하여, 이미지 생성을 위한 normal 프롬프트문을 생성하는 메서드

        Parameters:
        - text_json_analysis(Dict[str, Any]): _extract_text_features_via_gpt 메서드를 이용해 분석된, 챕터 요약문의 핵심 요소

        Returns:
        - Tuple[str, int, int]: 생성된 이미지 프롬프트 / 입력 토큰 수 / 출력 토큰 수
        - normal prompt문 생성에 실패하는 경우 None 리턴
        '''
        #===== GPT 프롬프트 엔지니어링: START =====
        tips = """
        1. First, translate the given JSON data into English if it contains Korean.
        2. Keep the prompts simple and concise, around 200 words. A simple prompt means it is easy for a child to understand.
        3. Focus on describing a single scene or snapshot, as Midjourney does not understand the concept of time or sequence.
        4. Do not use metaphors and comparisons in descriptions, as they can confuse Midjourney. Use raw, literal descriptions instead.
        5. When describing objects or structures, if the names are not common or widely recognized, use simpler terms or more general words to express them. For example, refer to "New York Times Square" as "a bustling city district" and "Coca-Cola" as "a carbonated beverage."
        6. Emphasizing adjectives like "extra," "super," or "extremely" does not have a significant effect and can distract Midjourney from the main subject.
        7. Stick to using "and" as the primary conjunction, as other conjunctions like "furthermore," "moreover," or "while" can confuse Midjourney.
        8. Always include a clear subject in each clause or sentence, as Midjourney may not understand clauses without a subject. Repeat the subject (e.g., "The enchantress") instead of using pronouns like "she" or "her."
        9. Avoid using negative conjunctions like "but," as Midjourney may not understand the contrasting meaning between clauses. Instead, use commas to separate independent adjectives or phrases.
        """

        rules_v4 = """
        1. Based on the given JSON data, create a prompt in the form of combined noun phrases.
        2. Descriptions of objects containing text(e.g. newspapers, signs, banners, etc.) should not be reflected in the prompt.
        """

        assistant_description = f"""You are an AI assistant specializing in generating text prompts for Midjourney, an AI-based image generation tool. \
        Your task is to create unique and detailed prompts based on the JSON data written mainly in Korean. \
        When creating prompts, keep the following general tips and rules in mind:

        Tips: {tips}
        Rules: {rules_v4}
        """

        user_message  = f"""Write a prompt in English for image generation within a maximum of 200 words in JSON format. \
        The JSON key for the generated image prompts should be 'image_prompt':\n{text_json_analysis}\n
        """
        #===== GPT 프롬프트 엔지니어링: END =====

        retry = 0
        step2_input_tokens = 0
        step2_output_tokens = 0

        while retry < self.number_of_retries:
            retry += 1
            try:
                response = self.openai_client.chat.completions.create(
                    model = self.gpt_model_name,
                    temperature = self.prompt_temperature,
                    max_tokens = 1000,
                    messages = [
                        {
                            "role": "system",
                            "content": assistant_description
                        },
                        {
                            "role": "user",
                            "content": user_message
                        }
                    ],
                    response_format = {
                        "type": "json_object"
                    },
                )

                ''' "finish_reason"에 따른 예외처리 '''
                # 최대 토큰 수를 초과한 경우
                if response.choices[0].finish_reason == "length":
                    print("토큰 개수가 초과되었습니다. normal 프롬프트문 재생성이 필요합니다.")
                    return None
                # OpenAI 정책에 의해 결과가 필터링된 경우
                elif response.choices[0].finish_reason == "content_filter":
                    print("생성된 프롬프트문이 OpenAI 정책에 따라 필터링되었습니다. normal 프롬프트문 재생성이 필요합니다.")
                    return None
                else:
                    result = response.choices[0].message.content
                    json_analysis = json.loads(result)
                    generated_image_prompt = json_analysis["image_prompt"]

                    # input(prompt_tokens) / output(completion_tokens) 획득
                    step2_input_tokens += response.usage.prompt_tokens
                    step2_output_tokens += response.usage.completion_tokens

                    return generated_image_prompt, step2_input_tokens, step2_output_tokens
            except Exception as e:
                print(f"retry={retry}: 요약문의 주요 요소를 이용하여 normal 프롬프트문 생성을 재시도합니다. 에러: {e}")
                continue
        raise Exception("수차례 재시도했음에도 normal 프롬프트문 생성에 실패했습니다.")


    def _generate_dry_prompt_based_on_text_features_via_gpt(self, text_json_analysis: Dict[str, Any]) -> Optional[Tuple[str, int, int]]:
        '''
        챕터 요약문으로부터 분석된 핵심 요소(JSON format)를 이용하여, 이미지 생성을 위한 dry 프롬프트문을 생성하는 메서드
        - dry prompt: 주어진 정보만 활용하며, 별도의 부가적인 묘사 및 미사여구를 추가하지 않은 프롬프트문

        Parameters:
        - text_json_analysis(Dict[str, Any]): _extract_text_features_via_gpt 메서드를 이용해 분석된, 챕터 요약문의 핵심 요소

        Returns:
        - Tuple[str, int, int]: 생성된 이미지 프롬프트 / 입력 토큰 수 / 출력 토큰 수
        - dry prompt문 생성에 실패하는 경우 None 리턴
        '''
        #===== GPT 프롬프트 엔지니어링: START =====
        tips = """
        1. First, translate the given JSON data into English if needed.
        2. Keep the prompts simple and concise, around 200 words. A simple prompt means it is easy for a child to understand.
        3. Focus on describing a single scene or snapshot, as Midjourney does not understand the concept of time or sequence.
        4. Do not use metaphors and comparisons in descriptions, as they can confuse Midjourney. Use raw, literal descriptions instead.
        5. When describing objects or structures, if the names are not common or widely recognized, use simpler terms or more general words to express them. For example, refer to "New York Times Square" as "a bustling city district" and "Coca-Cola" as "a carbonated beverage."
        6. Emphasizing adjectives like "extra," "super," or "extremely" does not have a significant effect and can distract Midjourney from the main subject.
        7. Stick to using "and" as the primary conjunction, as other conjunctions like "furthermore," "moreover," or "while" can confuse Midjourney.
        8. Always include a clear subject in each clause or sentence, as Midjourney may not understand clauses without a subject. Repeat the subject (e.g., "The enchantress") instead of using pronouns like "she" or "her."
        9. Avoid using negative conjunctions like "but," as Midjourney may not understand the contrasting meaning between clauses. Instead, use commas to separate independent adjectives or phrases.
        """

        dry_rules = """
        1. Based on the given JSON data, create a prompt in the form of combined noun phrases.
        2. Use only the JSON data when creating the prompt. Do not utilize any information other than the given data.
        3. If the given JSON data contains objects with text(e.g., newspapers, signs, banners, etc.), exclude them when creating the prompt.
        """

        assistant_description = f"""You are an AI assistant specializing in generating text prompts for Midjourney, an AI-based image generation tool. \
        Your task is to create prompts based on the JSON data written mainly in Korean. \
        When creating prompts, keep the following general tips and rules in mind:

        Tips: {tips}
        Rules: {dry_rules}
        """

        user_message  = f"""Write a prompt in English for image generation within a maximum of 200 words in JSON format. \
        The JSON key for the generated image prompts should be 'image_prompt':\n{text_json_analysis}\n
        """
        #===== GPT 프롬프트 엔지니어링: END =====

        retry = 0
        dry_input_tokens = 0
        dry_output_tokens = 0

        while retry < self.number_of_retries:
            retry += 1
            try:
                response = self.openai_client.chat.completions.create(
                    model = self.gpt_model_name,
                    temperature = self.dry_temperature,
                    max_tokens = 1000,
                    messages = [
                        {
                            "role": "system",
                            "content": assistant_description
                        },
                        {
                            "role": "user",
                            "content": user_message
                        }
                    ],
                    response_format = {
                        "type": "json_object"
                    },
                )

                ''' "finish_reason"에 따른 예외처리 '''
                # 최대 토큰 수를 초과한 경우
                if response.choices[0].finish_reason == "length":
                    print("토큰 개수가 초과되었습니다. dry 프롬프트문 재생성이 필요합니다.")
                    return None
                # OpenAI 정책에 의해 결과가 필터링된 경우
                elif response.choices[0].finish_reason == "content_filter":
                    print("생성된 프롬프트문이 OpenAI 정책에 따라 필터링되었습니다. dry 프롬프트문 재생성이 필요합니다.")
                    return None
                else:
                    result = response.choices[0].message.content
                    json_analysis = json.loads(result)
                    generated_dry_prompt = json_analysis["image_prompt"]

                    # input(prompt_tokens) / output(completion_tokens) 획득
                    dry_input_tokens += response.usage.prompt_tokens
                    dry_output_tokens += response.usage.completion_tokens

                    return generated_dry_prompt, dry_input_tokens, dry_output_tokens
            except Exception as e:
                print(f"retry={retry}: 요약문의 주요 요소를 이용하여 dry 프롬프트문 생성을 재시도합니다. 에러: {e}")
                continue
        raise Exception("수차례 재시도했음에도 dry 프롬프트문 생성에 실패했습니다.")


    ''' 2024-08-19 추가 '''
    def _check_containing_specified_elements(self, text: str) -> Optional[Tuple[bool, int, int]]:
        '''
        생성된 프롬프트문에 제외하고 싶은 특정 요소(e.g. 국기, 한국적인 요소)가 포함되어 있는지 체크하는 메서드

        Parameters:
        - text: 생성된 프롬프트문

        Returns:
        - Tuple[bool, int, int]: 특정 요소 포함 여부 / 입력 토큰 수 / 출력 토큰 수
        - 특정 요소의 포함 여부를 판단하기 어려운 경우 None을 리턴
        '''
        #===== GPT 프롬프트 엔지니어링: START =====
        criteria = """
        1. Check whether the given text includes flags or descriptions related to them.
        2. Check for traditional or historical objects from Korea, locations or objects that can be identified as Korean, or individuals of Korean nationality.
        """

        assistant_description = f"""You are an AI assistant specializing in evaluating the given text for culturally significant elements. Your primary tasks are as follows:

        Criteria: {criteria}
        """

        user_message = f"""Refer to the criteria above to evaluate the given text below. \
        If the text satisfies any one of the provided criteria, return true. If none of the criteria are satisfied, return false.
        Return the result in JSON format and the JSON key for result should be 'check_condition'

        Text: {text}
        """
        #===== GPT 프롬프트 엔지니어링: END =====

        retry = 0
        check_condition_input_tokens = 0
        check_condition_output_tokens = 0

        while retry < self.number_of_retries:
            retry += 1
            try:
                response = self.openai_client.chat.completions.create(
                    model = self.gpt_model_name,
                    temperature = self.common_temperature,
                    max_tokens = 1000,
                    messages = [
                        {
                            "role": "system",
                            "content": assistant_description
                        },
                        {
                            "role": "user",
                            "content": user_message
                        }
                    ],
                    response_format = {
                        "type": "json_object"
                    },
                )

                ''' "finish_reason"에 따른 예외처리 '''
                # 최대 토큰 수를 초과한 경우
                if response.choices[0].finish_reason == "length":
                    print("토큰 개수가 초과되었습니다.")
                    return None
                # OpenAI 정책에 의해 결과가 필터링된 경우
                elif response.choices[0].finish_reason == "content_filter":
                    print("생성된 결과가 OpenAI 정책에 따라 필터링되었습니다.")
                    return None
                else:
                    result = response.choices[0].message.content
                    json_analysis = json.loads(result)
                    check_condition = json_analysis["check_condition"]

                    # input(prompt_tokens) / output(completion_tokens) 획득
                    check_condition_input_tokens += response.usage.prompt_tokens
                    check_condition_output_tokens += response.usage.completion_tokens

                    return check_condition, check_condition_input_tokens, check_condition_output_tokens
            except Exception as e:
                print(f"retry={retry}: 생성된 프롬프트문에서, 제외 요소의 포함 여부 체크를 재시도합니다. 에러: {e}")
                continue
        raise Exception("수차례 재시도했음에도 제외 요소의 포함 여부 체크에 실패했습니다.")


    def select_best_prompt_via_gpt(self, text: str) -> Optional[Tuple[str, int, int]]:
        '''
        생성된 여러 프롬프트문 가운데, 최적의 프롬프트문을 선택하는 메서드

        Parameters:
        - text(str): 프롬프트문 생성에 사용되는 텍스트(주로 챕터 요약문)

        Returns:
        - Optional[Tuple[str, int, int]]: 최적의 프롬프트문 / 입력 토큰 수 / 출력 토큰 수
        - 최적의 프롬프트문 선택에 실패하는 경우 None을 리턴
        '''
        #===== GPT 프롬프트 엔지니어링: START =====
        criteria = """
        The clarity of scene description, consistency of mood and style, visually appealing and detailed depiction, and a variety of visual elements. \
        """

        assistant_description = f"""You are an AI assistant specializing in evaluating text prompts for image generation. \
        Your task is to choose the best prompt among several candidates. For evaluations, consider the following criteria: \n{criteria}\n
        """
        #===== GPT 프롬프트 엔지니어링: END =====

        # 챕터 요약문으로부터 주요 요소 추출
        # (챕터 요약문의 토큰 수가 많고, 거의 일정한 결과를 리턴하기 때문에 한 번만 추출 -> 에러 발생시에도 계속 사용함)
        text_json_analysis, step1_input_tokens, step1_output_tokens = self._extract_text_features_via_gpt(text=text)

        retry = 0
        total_input_tokens = step1_input_tokens
        total_output_tokens = step1_output_tokens

        while retry < self.number_of_retries:
            retry += 1
            try:
                # normal 프롬프트문 후보군 생성
                prompt_candidates = []

                for _ in range(self.number_of_prompts):
                    # normal 프롬프트문 생성
                    normal_prompt, step2_input_tokens, step2_output_tokens = self._generate_normal_prompt_based_on_text_features_via_gpt(text_json_analysis=text_json_analysis)
                    total_input_tokens += step2_input_tokens
                    total_output_tokens += step2_output_tokens

                    check_condition, check_condition_input_tokens, check_condition_output_tokens = self._check_containing_specified_elements(text=normal_prompt)
                    total_input_tokens += check_condition_input_tokens
                    total_output_tokens += check_condition_output_tokens

                    # 생성된 normal 프롬프트문에 미리 설정한 제외 요소가 없는 경우에만, append 수행하여 프롬프트문 후보군에 더함
                    if check_condition == False:
                        prompt_candidates.append(normal_prompt)

                # 생성된 모든 normal 프롬프트문이 제외 요소를 포함하고 있는 경우 -> dry prompt문 생성
                if len(prompt_candidates) == 0:
                    print("dry prompt문 생성을 시도합니다.")
                    dry_prompt, dry_input_tokens, dry_output_tokens = self._generate_dry_prompt_based_on_text_features_via_gpt(text_json_analysis=text_json_analysis)
                    total_input_tokens += dry_input_tokens
                    total_output_tokens += dry_output_tokens

                    # dry prompt에도 제외 요소가 있는지 체크하여, 이상 없으면 평가를 위한 단일 후보군으로 추가
                    check_condition_for_dry_prompt, check_condition_for_dry_input_tokens, check_condition_for_dry_output_tokens = self._check_containing_specified_elements(text=dry_prompt)
                    total_input_tokens += check_condition_for_dry_input_tokens
                    total_output_tokens += check_condition_for_dry_output_tokens

                    if check_condition_for_dry_prompt == False:
                        prompt_candidates.append(dry_prompt)

                # 평가할 프롬프트 후보군이 없는 경우(normal, dry쪽에서 모두 리턴값 없을 때), 재시도를 위해 continue 수행
                if not prompt_candidates:
                    print(f"retry={retry}: 적절한 프롬프트문 후보군이 생성되지 않았습니다. 재시도합니다.")
                    continue

                #===== GPT 프롬프트 엔지니어링: START =====
                user_message = f"""Choose the best prompt among several prompt candidates below. \
                Prompt Candidates: {prompt_candidates}

                Return the result in JSON format and the JSON key for the best prompt should be 'best_image_prompt.'
                """
                #===== GPT 프롬프트 엔지니어링: END =====

                response = self.openai_client.chat.completions.create(
                    model =  self.gpt_model_name,
                    temperature = self.common_temperature,
                    max_tokens = 1000,
                    messages = [
                        {
                            "role": "system",
                            "content": assistant_description
                        },
                        {
                            "role": "user",
                            "content": user_message
                        }
                    ],
                    response_format = {
                        "type": "json_object"
                    },
                )

                ''' "finish_reason"에 따른 예외처리 '''
                # 최대 토큰 수를 초과한 경우
                if response.choices[0].finish_reason == "length":
                    print("토큰 개수가 초과되었습니다. 프롬프트문 후보군의 재평가가 필요합니다.")
                    return None
                # OpenAI 정책에 의해 결과가 필터링된 경우
                elif response.choices[0].finish_reason == "content_filter":
                    print("생성된 결과가 OpenAI 정책에 따라 필터링되었습니다. 프롬프트문 후보군의 재평가가 필요합니다.")
                    return None
                else:
                    analysis = response.choices[0].message.content
                    json_analysis = json.loads(analysis)
                    best_image_prompt = json_analysis["best_image_prompt"]

                    # input(prompt_tokens) / output(completion_tokens) 획득
                    step3_input_tokens = response.usage.prompt_tokens
                    step3_output_tokens = response.usage.completion_tokens

                    total_input_tokens += step3_input_tokens
                    total_output_tokens += step3_output_tokens

                    return best_image_prompt, total_input_tokens, total_output_tokens
            except Exception as e:
                print(f"retry={retry}: 프롬프트문 후보군 가운데, 최적의 프롬프트문을 선택하지 못했습니다. 재시도합니다. 에러: {e}")
                continue
        raise Exception("수차례 재시도했음에도 최적의 프롬프트문 선택에 실패했습니다.")


    ''' 기록 유지 차원에서 남겨둠 '''
    # def _extract_text_features_via_claude(self, text):
    #     retry = 0
    #     while retry < self.number_of_retries:
    #         retry += 1

    #         rules = """
    #         1. From the given text, select the most important location to be used as the background for an image.
    #         2. Utilize the content of the text as much as possible to describe the location with a focus on visual elements. Express the description in the form of noun phrases.
    #         3. Select key elements that can be included in the analyzed most important location and describe them.
    #         4. Exclude people(figures) as key elements.
    #         5. Considering the atmosphere of the most important location selected above, suggest a 'painting style'.
    #         """

    #         json_schema = """
    #             {
    #                 "type": "object",
    #                 "properties": {
    #                     "most_important_location": {
    #                         "type": "string",
    #                         "description": "The most important location of the given text."
    #                     },
    #                     "location_description": {
    #                         "type": "array",
    #                         "description": "The description of the most important location. It would be expressed in the form of noun phrases."
    #                     },
    #                     "key_elements": {
    #                         "type": "array",
    #                         "description": "The list of key elments of the given text.",
    #                         "items": {
    #                             "type": "object",
    #                             "properties": {
    #                                 "key_element": {
    #                                     "type": "string",
    #                                     "description": "The name of key element."
    #                                 },
    #                                 "key_element_description": {
    #                                     "type": "string",
    #                                     "description": "The description of the key element."
    #                                 }
    #                             },
    #                             "required": ["key_element", "key_element_description"]
    #                         }
    #                     },
    #                     "painting_style": {
    #                         "type": "string",
    #                         "description": "The suggetsted painting style."
    #                     }
    #                 },
    #                 "required": ["most_important_location", "location_description", "key_elements", "painting_style"]
    #             }    
    #         """

    #         system_prompt = f"""You are an AI assistant specialized in analyzing text, mainly summaries of certain books. \
    #         Your task is to select the most important location and describe it along with key elements and suggesting a painting style. \
    #         For this analysis, keep the following rules in mind: \
    #         <rules>{rules}</rules>
    #         """

    #         user_message = f"""Select the most important location from the given text and describe it along with key elements and suggesting a painting style in JSON format. \
    #         <text>{text}</text>

    #         Return the result in the following JSON format. \
    #         <schema>{json_schema}</schema>
    #         """

    #         try:
    #             message = self.claude_client.messages.create(
    #                 # model="claude-3-haiku-20240307",
    #                 model="claude-3-5-sonnet-20240620",
    #                 max_tokens=1000,
    #                 temperature=self.feature_temperature,
    #                 system=system_prompt,
    #                 messages=[
    #                     {
    #                         "role": "user",
    #                         "content": [
    #                             {
    #                                 "type": "text",
    #                                 "text": user_message
    #                             }
    #                         ]
    #                     }
    #                 ]
    #             )
    #             result = message.content[0].text
    #             json_analysis = json.loads(result)

    #             return json_analysis
    #         except Exception as e:
    #             print(e)
    #     raise Exception("수차례 재시도했음에도 주요 요소 추출에 실패했습니다.")


    # def _generate_prompt_based_on_text_features_via_claude(self, json_analysis):
    #     retry = 0
    #     while retry < self.number_of_retries:
    #         retry += 1

    #         tips = """
    #         1. First, translate the given JSON data into English.
    #         2. Keep the prompts simple and concise, around 60 words. A simple prompt means it is easy for a child to understand.
    #         3. Focus on describing a single scene or snapshot, as Midjourney does not understand the concept of time or sequence.
    #         4. Do not use metaphors and comparisons in descriptions, as they can confuse Midjourney. Use raw, literal descriptions instead.
    #         5. When describing objects or structures, if the names are not common or widely recognized, use simpler terms or more general words to express them. For example, refer to "New York Times Square" as a "bustling city district" and "Coca-Cola" as a "carbonated beverage."
    #         6. Emphasizing adjectives like "extra," "super," or "extremely" does not have a significant effect and can distract Midjourney from the main subject.
    #         7. Stick to using "and" as the primary conjunction, as other conjunctions like "furthermore," "moreover," or "while" can confuse Midjourney.
    #         8. Always include a clear subject in each clause or sentence, as Midjourney may not understand clauses without a subject. Repeat the subject (e.g., "The enchantress") instead of using pronouns like "she" or "her."
    #         9. Avoid using negative conjunctions like "but," as Midjourney may not understand the contrasting meaning between clauses. Instead, use commas to separate independent adjectives or phrases.
    #         """
            
    #         rules_v4 = """
    #         1. Based on the given JSON data, create a prompt in the form of combined noun phrases.
    #         2. Descriptions of objects containing text(e.g. newspapers, signs, banners, etc.) should not be reflected in the prompt.
    #         """

    #         json_schema = """
    #             {
    #                 "type": "object",
    #                 "properties": {
    #                     "image_prompt": {
    #                         "type": "string",
    #                         "description": "The generated image prompt based on the givn JSON data."
    #                     }
    #                 },
    #                 "required": ["image_prompt"]
    #             }
    #         """

    #         system_prompt = f"""You are an AI assistant specializing in generating text prompts for Midjourney, an AI-based image generation tool. \
    #         Your task is to create unique and detailed prompts based on the JSON data written mainly in English or Korean. \
    #         When creating prompts, keep the following general tips and rules in mind: \
            
    #         <tips>{tips}</tips>
    #         <rules>{rules_v4}<rules>
    #         """

    #         user_message  = f"""Write a prompt in English for image generation within a maximum of 60 words using the given JSON data. \
    #         <json_analysis>{json_analysis}</json_analysis>

    #         Return the result in the following JSON format without any explanation. 'image_prompt' should be the key of JSON result.
    #         <json_schema>{json_schema}</json_schema>
    #         """

    #         try:
    #             message = self.claude_client.messages.create(
    #                 # model="claude-3-haiku-20240307",
    #                 model="claude-3-5-sonnet-20240620",
    #                 max_tokens=1000,
    #                 temperature=self.prompt_temperature,
    #                 system=system_prompt,
    #                 messages=[
    #                     {
    #                         "role": "user",
    #                         "content": [
    #                             {
    #                                 "type": "text",
    #                                 "text": user_message
    #                             }
    #                         ]
    #                     }
    #                 ]
    #             )
    #             result = message.content[0].text
    #             final_json_analysis = json.loads(result)
    #             generated_image_prompt = final_json_analysis["image_prompt"]

    #             return generated_image_prompt
    #         except Exception as e:
    #             print(e)
    #     raise Exception("수차례 재시도했음에도 프롬프트문 생성에 실패했습니다.")


    # def select_best_prompt_via_claude(self, text):
    #     retry = 0
    #     while retry < self.number_of_retries:
    #         retry += 1

    #         json_text_features = self._extract_text_features_via_claude(text)

    #         prompt_candidates = []
    #         for _ in range(self.number_of_prompts):
    #             prompt = self._generate_prompt_based_on_text_features_via_claude(json_text_features)
    #             prompt_candidates.append(prompt)

    #         system_prompt = """You are an AI assistant specializing in evaluating text prompts for image generation. \
    #         Your task is to choose the best prompt among several candidates for creating the most accurate and visually appealing image. \
    #         Consider clarity, conciseness, and how well the prompt captures the essential elements of the original text. \
    #         """

    #         json_schema = """
    #         {
    #             "type": "object",
    #             "properties": {
    #                 "best_image_prompt": {
    #                     "type": "string",
    #                     "description": "The best prompt among several candidates."
    #                 }
    #             },
    #             "required": ["best_image_prompt"]
    #         }
    #         """

    #         user_message = f"""Here are several candidate prompts for image generation based on the certin text. \
    #         <prompt_candiates>{prompt_candidates}</prompt_candidates>
            
    #         Select the best prompt by considering the following elements: \
    #         clarity of scene description, consistency of mood and style, visually appealing and detailed depiction, and a variety of visual elements.
            
    #         Return the result in the following JSON format withtout any explanation. \
    #         <json_schema>{json_schema}</json_schema>
    #         """

    #         try:
    #             message = self.claude_client.messages.create(
    #                 # model="claude-3-haiku-20240307",
    #                 model="claude-3-5-sonnet-20240620",
    #                 max_tokens=1000,
    #                 temperature=self.common_temperature,
    #                 system=system_prompt,
    #                 messages=[
    #                     {
    #                         "role": "user",
    #                         "content": [
    #                             {
    #                                 "type": "text",
    #                                 "text": user_message
    #                             }
    #                         ]
    #                     }
    #                 ],
    #             )
    #             result = message.content[0].text
    #             json_result = json.loads(result)
    #             best_image_prompt = json_result["best_image_prompt"]

    #             return best_image_prompt
    #         except Exception as e:
    #             print(e)
    #     raise Exception("수차례 재시도했음에도 최적의 프롬프트문 선택에 실패했습니다.")
#===============================================================================================


    def send_prompt_to_midjourney(self) -> None:
        '''
        Discord API를 이용하여 미드저니에 이미지 생성 프롬프트를 전달하는 메서드
        '''
        headers ={
            "authorization": self.authorization
        }

        # LLM으로부터 전달받은 프롬프트에 특수기호가 있거나, 이상하게 넘어올 가능성을 대비하는 차원
        prompt = " ".join(self.generated_prompt.split())
        prompt = re.sub(r"[^a-zA-Z0-9\s.,]+", "", prompt)

        # (2024-08-08)표지이미지를 사용하는 경우와, 그렇지 않은 경우를 분리
        if self.coverimage_url:
            option_value = str(prompt) + " --sref " + str(self.coverimage_url) + str(self.image_suffix) + str(self.suffix)
        else:
            option_value = str(prompt) + str(self.suffix)

        # json payload 정의
        payload = {
            "type": 2,
            "application_id": self.application_id,
            "guild_id": self.guild_id,
            "channel_id": self.channel_id,
            "session_id": self.session_id,
            "data": {
                "version": self.version,
                "id": self.id,
                "name": "imagine",
                "type": 1,
                "options": [{
                    "type": 3,
                    "name": "prompt",
                    "value": option_value
                }],
                "attachments": []
            }
        }

        # 프롬프트문을 최대 5회까지 재전송 시도
        for _ in range(5):
            r = requests.post("https://discord.com/api/v9/interactions", json=payload, headers=headers)
            if r.status_code == 204:
                print(f"프롬프트문이 정상적으로 미드저니에 전송되었습니다.")
                return
            else:
                print(f"프롬프트 전송 실패, 상태 코드: {r.status_code}, 응답: {r.text}")
        raise Exception("수차례 재시도했음에도, 프롬프트를 미드저니에 전송하지 못했습니다.")
#======================================================================
# 챕터 요약문을 이용해 프롬프트문을 생성하고, 이를 미드저니에 전달하는 클래스 : END
#======================================================================