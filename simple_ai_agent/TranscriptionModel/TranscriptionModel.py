from abc import ABC, abstractmethod
import logging
import numpy as np
import speech_recognition as sr
from scipy.io.wavfile import write


class TranscriptionModel(ABC):
    """
    語音轉文字 ABC
    """

    @staticmethod
    def create_model(model_type, **kwargs):
        """
        Init model

        Parameters:
        - model_type: model name
        - kwargs: model 參數

        Returns:
        - TranscriptionModel: 對應的模型實例
        """
        if model_type == "google":
            return GoogleSpeechModel(**kwargs)
        else:
            raise ValueError(f"不支援的model: {model_type}")

    @abstractmethod
    def transcribe(self, data, sample_rate):
        """
        聲音轉文字。

        Parameters:
        - data: NumPy 格式的音頻數據
        - sample_rate: 音頻採樣率

        Returns:
        - text: 轉錄結果字符串
        """
        pass


class GoogleSpeechModel(TranscriptionModel):
    """
    speech_recognition 的預設是google
    """

    def __init__(self, language="zh-TW", channel_count=2, sample_rate=44100):
        """
        初始化 GoogleSpeechModel。

        Parameters:
        - language: 語系'zh-TW'
        - channel_count: 音頻通道數:2
        - sample_rate: 取樣率
        """
        self.language = language
        self.channel_count = channel_count
        self.recognizer = sr.Recognizer()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("GoogleSpeechModel")
        self.sample_rate = sample_rate

    def transcribe(self, data, sample_rate, saving=False):
        """
        將錄音數據轉為文字。

        Parameters:
        - data: NumPy 格式的音頻數據
        - sample_rate: 音頻採樣率
        - saving: 是否要產生聲音檔案和文字檔案

        Returns:
        - text: 文字, 失敗是 None
        """
        try:
            # 將 NumPy 音訊數據轉為 SpeechRecognition 格式
            audio = sr.AudioData(
                np.concatenate(data, axis=0).tobytes(), sample_rate, self.channel_count
            )
            text = self.recognizer.recognize_google(audio, language=self.language)
            self.logger.info(f"Google Speech: {text}")

            if not saving:
                return text
            else:
                filename_prefix = "recording"
                text_filename = f"{filename_prefix}.txt"
                with open(text_filename, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"文字檔保存位置: {text_filename}")

                audio_filename = f"{filename_prefix}.wav"
                final_audio = np.concatenate(data).astype(np.int16)
                write(audio_filename, self.sample_rate, final_audio)
                print(f"錄音檔保存位置: {audio_filename}")

        except sr.UnknownValueError:
            self.logger.warning("Google Speech 無法識別語音內容。")
        except sr.RequestError as e:
            self.logger.error(f"Google Speech API 錯誤: {e}")
        return None
