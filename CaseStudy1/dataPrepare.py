from datasets import load_dataset,load_from_disk
from torchvision.transforms import v2
import torch

def dataPrepare(dataset_dir="/Volumes/DataHub/dataProcessed/bubu", size=(28,28), save_dir="",for_svd=False,for_CNN=False):
    dataset = load_from_disk(dataset_path=dataset_dir)
    print("Dataset Loaded")
    img=dataset["image"][0]

    common_transform = v2.Compose([
        v2.Resize(size),          
        v2.Grayscale(num_output_channels=1),               
     ])
 
    to_svd=v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32,scale=True),
               v2.Lambda(lambda x:x.to("mps")),
               v2.Lambda(lambda x:torch.flatten(x).to("cpu",dtype=torch.float32)),
        ])

    to_cnn=v2.Compose([
                v2.PILToTensor(),
            ])

    dataset = dataset.map(lambda x:{"image":common_transform(x["image"])}, 
                         batched=True,  
                         batch_size=512)
    
    if for_CNN:
        datasetCNN = dataset.map(lambda x:{"image":to_cnn(x["image"])}, 
                         batched=True,  
                         batch_size=512)
        datasetCNN.set_format(type="torch", columns=["image"])
        datasetCNN.save_to_disk(save_dir+"/datasetCNN")
        print(f"Dataset saved at {save_dir}"+"datasetCNN")
    elif for_svd:
        
        datasetSVD = dataset.map(lambda x:{"image":to_svd(x["image"])}
                        )
        datasetSVD.set_format(type="torch", columns=["image"])
        datasetSVD.save_to_disk(save_dir+"/datasetSVD")
        print(f"Dataset saved at {save_dir}"+"/datasetSVD")
    else:
        img_processed=dataset["image"][5399]
        img_processed.show()




    
    # dataset.save_to_disk(f"{save_dir}/dataset")
    # print(f"Dataset saved at {save_dir}")
    return



if __name__=="__main__":
    dataPrepare(dataset_dir="/Volumes/DataHub/dataProcessed/12",size=(100,100),save_dir="/Volumes/DataHub/dataProcessed/12",for_svd=True)

