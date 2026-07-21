
from Weft.utils.exceptions import WeftException
from Weft.utils.zipextracter import extract_zip

zip_path = "Data/Raw Data/reddyranbeer openAI Data.zip"
extract_to = "Data/Extracted Data/"


def extract_data_from_zip(zip_path, extract_to):
    try:
        extract_zip(zip_path, extract_to)
        print(f"Data extracted successfully to {extract_to}")
    except WeftException as e:
        print(f"An error occurred while extracting data: {e}")


if __name__ == "__main__":
    extract_data_from_zip(zip_path, extract_to)
