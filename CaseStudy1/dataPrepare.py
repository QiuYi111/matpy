from datasets import load_dataset
from torchvision.transforms import v2
import torch

def dataPrepare(dataset_name="mertcobanov/animals", size=(600,600), save_dir=""):
    dataset = load_dataset(dataset_name, split="train")
    transform = v2.Compose([
        v2.Resize(size),          
        v2.Grayscale(num_output_channels=1),  
        v2.ToImage(),             
        v2.ToDtype(torch.float32, scale=True), 
    ])
    

    dataset = dataset.map(lambda x: {"image": transform(x["image"])}, 
                         batched=True,  
                         batch_size=128)
    
   
    dataset.save_to_disk(f"{save_dir}/dataset")
    print(f"Dataset saved at {save_dir}")
    return



if __name__=="__main__":
    dataPrepare(save_dir="/Volumes/DataHub/dataProcessed")

