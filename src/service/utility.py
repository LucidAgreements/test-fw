import logging
import os

logger = logging.getLogger('utility')


def get_files(file_root, files_to_filter) -> list:
    logger.info("getting audio files")
    file_list = []

    for root, _, files in os.walk(file_root):
        for file in files:
            file_path = os.path.join(root, file)
            file_path = os.path.abspath(file_path)
            file_list.append(file_path)

    return [file for file in file_list if file.lower().endswith(files_to_filter)]


def clean_all_line_breaks(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    return text


def clean_double_spaces(text: str) -> str:
    cleaned_text = ' '.join(text.split())
    return cleaned_text
