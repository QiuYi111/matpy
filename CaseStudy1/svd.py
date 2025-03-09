import torch

def svd(dataset_dir,):
    trainset=torch.load(dataset_dir)
    trainset=trainset-torch.mean(trainset,dim=1,keepdim=True)
    U, S, Vh = torch.linalg.svd(trainset,False)
    torch.save(U,dataset_dir+"U.pt")

