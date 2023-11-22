import zipfile

if __name__ == "__main__":
    with zipfile.ZipFile("/scratch/tw2672/Dataset_Student.zip", 'r') as zip_ref:
        zip_ref.extractall("/scratch/tw2672/dataset")