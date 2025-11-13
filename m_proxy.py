import json
import os
from mitmproxy import http


def _load_config(config_filename: str):
    '''
    JSON 파일에서 configuration을 로드하는 메서드
    
    Parameters:

    Returns:

    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, config_filename)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 '{config_path}'에서 찾을 수 없습니다.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "output_img_folder" not in config:
        raise KeyError("설정 파일에서 'output_img_folder'키를 찾을 수 없습니다.")

    return config


def response(flow: http.HTTPFlow) -> None:
    '''
    HTTP 응답이 이미지인 경우, 이를 지정된 폴더에 저장하는 메서드
    '''
    try:
        config = _load_config("config_image_downloader.json")
        output_img_folder = config["output_img_folder"]
    except (FileNotFoundError, KeyError) as e:
        print(e)
        return

    if flow.response.headers.get("content-type", "").startswith("image/"):
        # 별도 파일명 지정
        '''
        이미지의 url이 "https://cdn.midjourney.com/75156d4c-1fa1-407b-aab0-80470f831765/0_2.png"인 경우,
        다운받는 임시 파일명은 "75156d4c-1fa1-407b-aab0-80470f831765_2.png"로 지정
        '''
        url_parts = flow.request.url.split("/")
        custom_id = url_parts[-2]
        candidate_num = url_parts[-1]
        filename = f"{custom_id}_{candidate_num.split('_')[-1]}"

        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_img_folder, filename)
        with open(image_path, "wb") as f:
            f.write(flow.response.content)