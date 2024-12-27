import os
from dotenv import load_dotenv
from simple_ai_agent.core import AudioAI

# 加載 .env 配置
load_dotenv()

if __name__ == "__main__":
    audio_ai = AudioAI()
    audio_ai.start()
