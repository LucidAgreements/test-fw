import json
import logging
import os.path

import torch
from faster_whisper import WhisperModel

logger = logging.getLogger('service_transcriber')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s][%(name)s] %(message)s', force=True)


class FasterWhisperService:
    def __init__(self):
        logger.info("init")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = "isLucid/faster-whisper-large-v2"
        self.model = None

    @staticmethod
    def __seconds_to_milliseconds(seconds: float) -> int:
        return int(round(seconds * 1000))

    def load(self):
        logger.info("loading model")
        self.model = WhisperModel(self.model_size,
                                  download_root="downloads",
                                  device=str(self.device),
                                  compute_type="auto")

    def transcribe(self, file_path: str) -> dict:
        logger.info("transcribing file: " + file_path)
        if os.path.exists(file_path):
            results = []
            segments, info = self.model.transcribe(
                file_path,
                language="lt",
                hallucination_silence_threshold=2.0,
                vad_filter=True)

            duration = self.__seconds_to_milliseconds(info.duration)
            duration_after_vad = self.__seconds_to_milliseconds(info.duration_after_vad)

            for segment in segments:
                json_format = {"start": self.__seconds_to_milliseconds(segment.start),
                               "end": self.__seconds_to_milliseconds(segment.end),
                               "text": segment.text.strip()}
                results.append(json_format)

            file_name = file_path.split("/")

            with open("/Users/martynastoleikis/Desktop/Project/test-fw/transcribed_files/{}.json".format(
                    file_name[len(file_name) - 1]), "w", encoding="utf-8") as outfile:
                json.dump(results, outfile, ensure_ascii=False)

            return dict(text=results, duration=duration, duration_after_vad=duration_after_vad, language=info.language)
        else:
            return dict(text=[], duration=0, duration_after_vad=0, language=None)

    def release(self):
        logger.info("releasing")
        if self.model:
            del self.model
        torch.cuda.empty_cache()
