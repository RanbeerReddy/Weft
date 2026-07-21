from zipfile import ZipFile


# Open the Zip file
def extract_zip(zip_path, extract_to):
    with ZipFile(zip_path, "r") as z_ref:
        z_ref.extractall(extract_to)
