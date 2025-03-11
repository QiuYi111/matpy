import torch
from datasets import load_from_disk
from PIL import Image
from torchvision.transforms import v2
class svd():
    def __init__(self,r,data_num=5400,label_num=90,dataset_dir="/Volumes/DataHub/dataProcessed/datasetSVD"):
        print("SVD Initializing!")
        self.data_num=data_num
        self.label_num=label_num
        self.dataset=load_from_disk(dataset_dir)
        self.r=r
        self.common_trans=v2.Compose([
        v2.ToDtype(torch.uint8),
        v2.Resize((600,600)),          
        v2.Grayscale(num_output_channels=1),
        v2.PILToTensor()              
     ])
        print("SVD Initialized")

    def get_eigenMatrix(self):
        X=[]
        i=0
        for sample in self.dataset['image']:
            sample=sample.to("mps")
            if i==0:
                print("Start Loading!")
                print(sample.shape)
            elif i%100==0:
                print(i/self.data_num)
            sample=torch.reshape(sample,(360000,))
            X.append(sample)
            i+=1
            if i >= self.data_num:
                break
        print("finish building",i,"columns' X")
        self.X=torch.stack(X).to(dtype=torch.float32)
        self.X_loaded=True
        torch.save(self.X,"/Volumes/DataHub/dataProcessed/X.pt")
        print("SVD start! X shape is ",X.shape)

        U,S,Vh=torch.linalg.svd(X,full_matrices=False)

        print("SVD Completed!")

        torch.save(U,"/Volumes/DataHub/dataProcessed/eigenMatrix.pt")

        print("U matrix Saved!")

        return U
    
    def load_eigenMatrix(self,U_dir="/Volumes/DataHub/dataProcessed/eigenMatrix.pt"):
        U=torch.load(U_dir)
        self.U=U.to("mps")
    
    def get_selected_U(self):
        self.Ur=self.U[:,:self.r]
        print("Build ",self.r,"columns Ur!")
    
    def load_image(self,img_path):
        img=Image.open(img_path)
        img_tensor=self.common_trans(img)
        img_tensor=img_tensor.to("mps")
        img=torch.reshape(img_tensor,(36000,))
        return img
    
    def show_img(self,img_tensor):
        img_tensor=torch.reshape(img_tensor,(600,600))
        img=v2.ToPILImage(img_tensor)
        img.show()


    def img_rebuild(self,img_path):
        img=self.load_image(img_path)
        
        re_img=self.Ur @ (self.Ur.T @ img)
        self.show_img(re_img)

    def get_alpha(self,img_path):
        img=self.load_image(img_path)
        alpha=self.Ur.T @ img
        return alpha
    
    def label_distro(self,label):
        label_idx=self.dataset["label"].index(label)
        if self.X_loaded==False:
            X=torch.load("/Volumes/DataHub/dataProcessed/X.pt")
            self.X_loaded=True
        selected_X=X[:,label_idx]
        Alpha=self.Ur.T @ selected_X
        Alpha_mean=Alpha.mean(dim=0)
        Alpha_delta=Alpha.var()
        distances=torch.norm(Alpha-Alpha_mean,2,dim=1)
        max_distance=torch.max(distances)
        return {
            "mean": Alpha_mean,
            "var": Alpha_delta,
            "range":max_distance
        }

    def recognize(self,img_path):
        img=self.load_image(img_path)
        alpha=self.Ur.T @ img
        answer=[]
        for label in range(self.label_num):
            prob=0
            label_distro=self.label_distro(label)
            distance=torch.norm(alpha-label_distro["mean"],2)
            sig=max(3*label_distro["var"],label_distro["range"])
            if distance <= sig:
                prob=distance/label_distro["range"]
            answer.append(prob)
        most_prob_label=answer.index(max(answer))
        return {
            "answer":answer,
            "most_prob_label":most_prob_label
        }
    
if __name__ =="__main__":
    SVD=svd(r=100)
    SVD.get_selected_U()
    SVD.load_eigenMatrix
    SVD.img_rebuild("/Desktop/image.jpg")
    

