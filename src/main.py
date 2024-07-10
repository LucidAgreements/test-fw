import logging

from src.fw_service import FasterWhisperService
from src.service.utility import get_files


logger = logging.getLogger('lunching main')

def main():
    faster_whisper_service = FasterWhisperService()
    faster_whisper_service.load()

    for file in get_files(file_root="/Users/martynastoleikis/Downloads/Archive1", files_to_filter=(".wav", ".mp3")):
        faster_whisper_service.transcribe(file)

    faster_whisper_service.release()


if __name__ == "__main__":
    main()
