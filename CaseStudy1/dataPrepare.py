from datasets import load_dataset
from torchvision.transforms import v2
import torch

def dataPrepare(dataset_name="mertcobanov/animals",size=(600,600),save_dir=""):
    dataset=load_dataset(dataset_name,split="train")
    print("Dataset Length:",len(dataset["image"]))
    transform=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8,scale=True),
        v2.Resize(size=size),
        v2.Grayscale(),
        v2.ToDtype(torch.float32,scale=True),
        v2.Lambda(lambda x:torch.torch.flatten(x))
    ]
    )
    dataset=dataset.map(lambda x: {"image": transform(x["image"])})
    all_images=[sample["image"] for sample in dataset]
    imgTensor=torch.stack(all_images,dim=0)
    print("Image Set Size: ",imgTensor.shape)
    dataset.save_to_disk(save_dir+"/dataset")
    torch.save(imgTensor,save_dir+"/tensor.pt")
    print("Dataset Prepare Complete! Saved at ",save_dir)


if __name__=="__main__":
    dataPrepare(save_dir="./dataProcessed")

