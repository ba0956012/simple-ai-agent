import os
from dotenv import load_dotenv
from simple_ai_agent.core import AudioAI

if __name__ == "__main__":
    load_dotenv()

    sample_rate = int(os.getenv("SAMPLE_RATE", 44100))
    chunk = int(os.getenv("CHUNK", 32000))
    silence_threshold = int(os.getenv("SILENCE_THRESHOLD", 10))
    silence_duration = int(os.getenv("SILENCE_DURATION", 2))
    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.8))

    camera_index = int(os.getenv("CAMERA_INDEX", 0))
    mic_index = int(os.getenv("MIC_INDEX", 0))

    person_detector = os.getenv("PERSON_DETECTOR", "YOLOv5")
    person_detector_model = os.getenv(
        "PERSON_DETECTOR_MODEL", "./simple_ai_agent/PersonDetector/model/yolov5n.onnx"
    )
    person_conf_thres = float(os.getenv("PERSON_CONF_THRES", 0.5))
    person_iou_thres = float(os.getenv("PERSON_IOU_THRES", 0.5))
    person_input_size = tuple(
        map(int, os.getenv("PERSON_INPUT_SIZE", "320,320").split(","))
    )

    transcribe_audio_model = os.getenv("TRANSCRIBE_AUDIO_MODEL", "google")

    audio_to_text_keywords = os.getenv(
        "AUDIO_TO_TEXT_KEYWORDS", "語音轉文字,語音翻譯,轉換文字,STT"
    ).split(",")
    play_speaker_sound_keywords = os.getenv(
        "PLAY_SPEAKER_SOUND_KEYWORDS", "播放嗽叭,播放音樂,喇叭聲音"
    ).split(",")
    detect_people_count_keywords = os.getenv(
        "DETECT_PEOPLE_COUNT_KEYWORDS", "現場人數,人數檢測,檢查人數"
    ).split(",")

    keyword_matcher_mode = os.getenv("KEYWORD_MATCHER_MODE", "sBERT")

    # 初始化 AudioAI
    audio_ai = AudioAI(
        sample_rate=sample_rate,
        chunk=chunk,
        silence_threshold=silence_threshold,
        silence_duration=silence_duration,
        similarity_threshold=similarity_threshold,
        person_detector=person_detector,
        person_detector_model=person_detector_model,
        person_conf_thres=person_conf_thres,
        person_iou_thres=person_iou_thres,
        person_input_size=person_input_size,
        transcribe_audio_model=transcribe_audio_model,
        audio_to_text_keywords=audio_to_text_keywords,
        play_speaker_soundt_keywords=play_speaker_sound_keywords,
        detect_people_countt_keywords=detect_people_count_keywords,
        keyword_matcher_mode=keyword_matcher_mode,
        camera_index=camera_index,
        mic_index=mic_index
    )

    # 啟動系統
    audio_ai.start()
