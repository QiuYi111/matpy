import torch
from datasets import load_from_disk

dataset=load_from_disk("/Volumes/DataHub/dataProcessed/datasetSVD")

print("Dataset Loaded from disk")

X=[torch.reshape(sample["image"],(36000,)) for sample in dataset]

torch.stack(X)

print("X matrix built")

U,S,Vh=torch.linalg.svd(X)

print("SVD Completed!")

torch.save(U,"/Volumes/DataHub/dataProcessed/eigenMatrix.pt")

print("U matrix Saved!")
