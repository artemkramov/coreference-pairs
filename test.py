from os import listdir
from os.path import isfile, join
import io
from models.cluster_model import CoreferenceModel

# Get list of files to examine
directory_test = "test"
files = [join(directory_test, filename) for filename in listdir(directory_test) if
         isfile(join(directory_test, filename))]

# Init cluster model
coreferenceModel = CoreferenceModel()

# Loop through each file
for filename in files:
    file = io.open(filename, mode="r", encoding="utf-8")
    raw_text = file.read()
    coreferenceModel.find_coreference_pairs(raw_text)
    file.close()
