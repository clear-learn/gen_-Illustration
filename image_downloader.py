import cv2
import easyocr
import json
import os
import time
import torch

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait

from get_custom_id import GetCustomId


class ImageDownloader:
    def __init__(self, get_custom_id: GetCustomId):
        self.get_custom_id = get_custom_id
        self.custom_id = get_custom_id.custom_id

        self.use_gpu: bool = torch.cuda.is_available()
        self.output_img_folder: str = ""
        self.easyocr_languages: list[str] = []

        self._config_initialization()
        self.image_url, self.local_filepath = self.capture_image_by_web_scraping()


    def _config_initialization(self):
        '''
        JSON 파일에서 configuration을 로드하는 메서드

        Raises:
        - FileNotFoundError: 설정 파일이 존재하지 않을 때 발생
        '''
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config_image_downloader.json")

        # 파일 유효성 체크
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            self.output_img_folder = config["output_img_folder"]
            self.easyocr_languages = config["easyocr_languages"]
        else:
            raise FileNotFoundError(f"설정을 '{config_path}'에서 로드할 수 없습니다.")


    def _detect_characters_using_easyocr(self, filename: str) -> bool:
        '''
        EasyOCR를 이용해 주어진 이미지 파일 대상으로 OCR을 수행하는 메서드

        Parameters:
        - filename: 이미지 파일 경로

        Returns:
        - bool: OCR을 수행하여 텍스트가 감지되면 TRUE, 감지되지 않는다면 FALSE를 리턴 
        '''
        detected_text = set()

        for lang in self.easyocr_languages:
            # 여러 언어를 한꺼번에 설정할 수 없어서, 루프문을 통해 결과를 합침
            reader = easyocr.Reader([lang], gpu=self.use_gpu)
            # "detail" 파라미터를 설정하지 않으면 "추츨된 글자의 꼭지점 좌표(top-left, top-right, bottom-right, bottom-left) / 추출된 글자 / 확률"이 리스트 형태로 리턴
            result = reader.readtext(filename, detail=0)

            if len(result) == 0:
                pass
            else:
                for character in result:
                    detected_text.add(character)

        detected_text_list = list(detected_text)

        return len(detected_text_list) > 0


    def _detect_faces_using_opencv(self, filename: str, show_bbox: bool=False) -> bool:
        '''
        OpenCV 기반으로, 주어진 이미지에서 얼굴 유무를 체크하는 메서드
        탐지된 얼굴에 대해 선택적으로 bounding box를 표시할 수 있음

        Parameters:
        - filename(str): 이미지 파일 경로
        - show_bbox(bool): 얼굴이 탐지된 경우의 bounding box 표시 유무

        Returns:
        - bool: 이미지에서 얼굴이 탐지되면 True, 탐지되지 않았으면 False 
        '''
        # Haar Cascade 분류기 선언
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # 이미지 로드
        loaded_image = cv2.imread(filename)
        if loaded_image is None:
            raise ValueError(f"에러: 이미지 '{filename}'을 로드할 수 없습니다.")

        # 이미지를 그레이스케일로 변환
        gray_scaled = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)

        # 얼굴 detection 수행
        faces = face_cascade.detectMultiScale(gray_scaled, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_detected = len(faces) > 0

        if face_detected and show_bbox:
            # 이미지에 bounding box 표시
            for (x, y, w, h) in faces:
                cv2.rectangle(loaded_image, (x,y), (x+w, y+h), (255,0,0), 2)

            if not os.path.exists(self.output_img_folder):
                os.makedirs(self.output_img_folder)

            # 결과 이미지를 output_folder에 저장
            base_filename = os.path.basename(filename)
            output_filename = os.path.join(self.output_img_folder, "detection_" + base_filename)
            cv2.imwrite(output_filename, loaded_image)

        return face_detected


    def capture_image_by_web_scraping(self):
        '''
        cdn 경로상의 이미지를 캡처하는 메서드

        Raises:
        - Exception: 웹 드라이버 작업 중 발생한 예외
        '''

        # Chrome 옵션 설정
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--ignore-certificate-errors")
        chrome_options.add_argument("--ignore-ssl-errors")
        chrome_options.add_argument("--window-size=1024,1024")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])  # 자동화 소프트웨어 메시지 비활성화
        chrome_options.add_experimental_option('useAutomationExtension', False)  # 자동화 확장 기능 사용하지 않음

        # 임시로 이미지를 저장하는 폴더 지정
        if not os.path.exists(self.output_img_folder):
            os.makedirs(self.output_img_folder)

        # 추출한 custom_id를 이용해 cdn 이미지 경로 지정 (미드저니는 디폴트로 이미지 4개 생성)
        base_url = "https://cdn.midjourney.com"
        image_urls_list = [f"{base_url}/{self.custom_id}/0_{i}.png" for i in range(4)]

        # ChromeDriver 실행
        driver = webdriver.Chrome(options=chrome_options)

        # navigator.webdriver 속성 숨기기
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            """
        })

        # 브라우저 지문 변경
        driver.execute_cdp_cmd("Network.setUserAgentOverride", {
            "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })

        img_count = 0
        for image_url in image_urls_list:
            if image_url is not None:

                # url에서 1024*1024 사이즈의 screenshot 수행
                driver.get(image_url)
                try:
                    WebDriverWait(driver, 20).until(
                        lambda d: d.execute_script("return document.readyState") == "complete"
                    )

                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    local_filepath = os.path.join(current_dir, f"{self.output_img_folder}/{self.custom_id}_{image_urls_list.index(image_url)}.png")
                    driver.save_screenshot(local_filepath)
                except Exception as e:
                    print(f"CDNs 크롤링 에러 : {e}")
                finally:
                    driver.quit()

                if os.path.exists(local_filepath):
                    text_detected = self._detect_characters_using_easyocr(local_filepath)
                    # face_detected = self._detect_faces_using_opencv(local_filepath, show_bbox=True)

                    # if text_detected or face_detected:
                    if text_detected:
                        # print("생성된 이미지에서 텍스트 또는 얼굴이 감지되었습니다.")
                        print("생성된 이미지에서 텍스트가 감지되었습니다.")
                        os.remove(local_filepath)
                        img_count += 1

                    return image_url, local_filepath

            if img_count == len(image_urls_list):
                print("생성된 이미지 4개에서 모두 텍스트 또는 얼굴이 탐지되었습니다. 처음부터 재시도해주세요.")

        # 모든 이미지를 순회한 후에도 결과가 리턴되지 않은 경우
        return None, None