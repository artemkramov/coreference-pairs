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

coreferenceModel.model_prepare.load_documents()
document = coreferenceModel.model_prepare.documents[list(coreferenceModel.model_prepare.documents.keys())[132]]

clusters = []

for cluster in document['clusters']:
    c = []
    for entity_id in document['clusters'][cluster]:
        c.append(document['entities'][entity_id])
    clusters.append(c)

coreferenceModel.print_clusters(clusters)
print("======================")
coreferenceModel.find_coreference_pairs_from_tokens(document['tokens'])

# # Loop through each file
# for filename in files:
#     file = io.open(filename, mode="r", encoding="utf-8")
#     raw_text = file.read()
#     coreferenceModel.find_coreference_pairs_from_text(raw_text)
#     file.close()
