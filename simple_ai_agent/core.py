import logging
import multiprocessing
import os

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from playsound import playsound

from simple_ai_agent.KeywordMatcher.KeywordMatcher import KeywordMatcher
from simple_ai_agent.PersonDetector.PersonDetector import PersonDetector
from simple_ai_agent.TranscriptionModel.TranscriptionModel import TranscriptionModel


class AudioAI:
    """
    AudioAI 系統，通過監聽音頻流並處理相關語音輸入。
    支持基於關鍵字的處理邏輯，包括語音轉文字、播放聲音和人數檢測。
    """

    def __init__(
        self,
        sample_rate=44100,
        chunk=32000,
        silence_threshold=10,
        silence_duration=2,
        similarity_threshold=0.8,
        person_detector="YOLOv5",
        person_detector_model="./simple_ai_agent/PersonDetector/model/yolov5n.onnx",
        person_conf_thres=0.5,
        person_iou_thres=0.5,
        person_input_size=(320, 320),
        transcribe_audio_model="google",
        audio_to_text_keywords="語音轉文字,語音翻譯,轉換文字, STT".split(","),
        play_speaker_soundt_keywords="播放嗽叭,播放音樂,喇叭聲音".split(","),
        detect_people_countt_keywords="現場人數,人數檢測,檢查人數".split(","),
        keyword_matcher_mode="sBERT",
        camera_index=0
    ):
        """
        初始化 AudioAI。
        從環境變數加載配置，初始化音頻參數和關鍵字處理邏輯。
        """

        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),  # 將日誌輸出到控制台
                logging.FileHandler(
                    "audio_ai.log", mode="a", encoding="utf-8"
                ),  # 保存到文件
            ],
        )

        self.logger = logging.getLogger("AudioAI")

        self.sample_rate = sample_rate  # 音頻採樣率
        self.chunk = chunk  # 音頻塊大小
        self.silence_threshold = silence_threshold  # 靜音判斷閾值
        self.silence_duration = silence_duration  # 靜音持續時間
        self.similarity_threshold = similarity_threshold  # 關鍵字的相似度閾值

        self.audio_data = []  # 用於存儲音頻數據的緩衝區
        self.recording = False  # 是否正在錄音
        self.text_mode = False
        self.silent_chunks = 0  # 靜音計數
        self.camera_index = camera_index # 相機index

        # 外部函數
        self.play_sound = playsound  # 播放聲音函數

        self.person_detector = PersonDetector(
            model_class=person_detector,
            model_path=person_detector_model,
            conf_thres=person_conf_thres,
            iou_thres=person_iou_thres,
            input_size=person_input_size,
        )

        self.transcribe_audio_model = TranscriptionModel.create_model(
            model_type=transcribe_audio_model
        )

        # 關鍵字與處理函數映射
        self.keywords = {
            self.process_audio_to_text: audio_to_text_keywords,
            self.play_speaker_sound: play_speaker_soundt_keywords,
            self.detect_people_count: detect_people_countt_keywords,
        }

        self.keyword_matcher = KeywordMatcher(model_name=keyword_matcher_mode)
        # 向量化關鍵字
        self.keyword_embeddings = {
            handler: self.keyword_matcher.encode_keywords(phrases)
            for handler, phrases in self.keywords.items()
        }

    def start(self):
        """
        啟動音頻處理系統，持續監聽麥克風輸入。
        """
        self.logger.info("系統啟動，開始監聽...")
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self._audio_callback,
                blocksize=self.chunk,
            ):
                while True:
                    pass
        except KeyboardInterrupt:
            self.logger.info("系統終止。")

    def _audio_callback(self, indata, frames, callback_time, status):
        """
        音頻數據回調函數，實時處理音頻塊。

        Parameters:
        - indata: 音頻數據
        - frames: 每塊的幀數
        - callback_time: 回調時間戳
        - status: 音頻狀態

        依據音量判斷是否開始錄音。
        """
        volume = np.linalg.norm(indata) * 10
        if volume > self.silence_threshold:
            self._start_recording(indata)
        elif self.recording:
            self._process_silence()

    def _start_recording(self, indata):
        """
        開始錄音

        Parameters:
        - indata: 新的音頻數據
        """
        if not self.recording:
            self.recording = True
            self.silent_chunks = 0
            self.audio_data.clear()
            self.logger.info("偵測到聲音，開始錄音...")
        self.audio_data.append((indata.flatten() * 32767).astype(np.int16))

    def _process_silence(self):
        """
        靜音檢測
        """
        self.silent_chunks += 1
        if self.silent_chunks > (self.sample_rate / self.chunk * self.silence_duration):
            self.logger.info("偵測到靜音，停止錄音...")
            self.recording = False
            self._analyze_audio()

    def _analyze_audio(self):
        """
        轉文字和處理功能
        """
        text = self.transcribe_audio_model.transcribe(
            self.audio_data, self.sample_rate, self.text_mode
        )
        if text and not self.text_mode:
            self.logger.info(f"音頻轉文字結果：{text}")
            for handler, embeddings in self.keyword_embeddings.items():
                scores = self.keyword_matcher.compute_similarity(text, embeddings)
                max_score = max(scores) if isinstance(scores, list) else scores.max()
                if max_score > self.similarity_threshold:
                    self.logger.info(f"相似關鍵字匹配，執行對應動作...")
                    handler()
                    break
            self.keyword_matcher.input_embedding = None
        elif text and self.text_mode:
            self.text_mode = False
        self.audio_data.clear()

    def process_audio_to_text(self):
        """
        執行 '語音轉文字' 的處理邏輯。
        """
        if not self.text_mode:
            self.logger.info("接下來的語音內容，會執行語音轉文字...")
            self.text_mode = True

    def play_speaker_sound(self):
        """
        執行 '播放嗽叭' 的處理邏輯。
        """
        self.logger.info("執行播放聲音...")
        self.play_sound("./simple_ai_agent/Hello.wav")

    def detect_people_count(self):
        """
        執行 '現場人數' 的處理邏輯，啟動獨立進程執行人數檢測。
        """
        self.logger.info("執行現場人數檢測...")
        process = multiprocessing.Process(target=self._run_person_detection())
        process.start()
        process.join(timeout=10)

    def _run_person_detection(self):
        """
        獨立進程中執行人數檢測邏輯。
        """
        count = self.person_detector.detect_persons(camera_index=self.camera_index)
        self.logger.info(f"檢測到現場人數：{count}")


if __name__ == "__main__":
    # 加載 .env 文件
    load_dotenv()

    # 提取參數
    sample_rate = int(os.getenv("SAMPLE_RATE", 44100))
    chunk = int(os.getenv("CHUNK", 32000))
    silence_threshold = int(os.getenv("SILENCE_THRESHOLD", 10))
    silence_duration = int(os.getenv("SILENCE_DURATION", 2))
    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.8))

    camera_index = int(os.getenv("CAMERA_INDEX", 0))
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
    )

    # 啟動系統
    audio_ai.start()
