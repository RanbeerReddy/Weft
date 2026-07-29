from Weft.config.settings import settings
from Weft.utils.exceptions import WeftException
from Weft.utils.logger import logger
from Weft.utils.zipextracter import extract_zip

zip_path = settings.RAW_DATA_ZIP
extract_to = settings.EXTRACTED_DATA_DIR


def extract_data_from_zip(zip_path, extract_to):
    try:
        extract_zip(zip_path, extract_to)
        logger.info(f"Data extracted successfully to {extract_to}")
    except WeftException as e:
        logger.error(f"An error occurred while extracting data: {e}")


if __name__ == "__main__":
    extract_data_from_zip(zip_path, extract_to)
