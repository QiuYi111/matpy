from datasets import load_from_disk
import torch
a=load_from_disk("/Volumes/DataHub/dataProcessed/datasetSVD")
print(a.features["label"].num_classes)