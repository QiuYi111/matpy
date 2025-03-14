import torch
from datasets import load_from_disk
from PIL import Image
from torchvision.transforms import v2
class svd():
    def __init__(self,r,label=0,data_num=5400,label_num=90,dataset_dir="/Volumes/DataHub/dataProcessed/datasetSVD"):
        print("SVD Initializing!")
        self.data_num=data_num
        self.label_num=label_num
        self.dataset=load_from_disk(dataset_dir)
        self.label=torch.tensor(self.dataset["label"])
        print("Loaded Dataset",self.dataset["image"][0].shape)
        self.r=r
        self.common_trans=v2.Compose([
        v2.Resize((28,28)),          
        v2.Grayscale(num_output_channels=1),
        v2.PILToTensor()              
     ])
        self.get_eigenMatrix(label)
        self.get_selected_U()

        print("SVD Initialized")

    def get_eigenMatrix(self,label):
        label_idx=(self.label==label).nonzero()
        print("Label Num",len(label_idx))        
        X=[]
        i=0
        for sample in self.dataset['image']:
            sample=sample.to("mps")
            if i==0:
                print("Start Loading!")
                print(sample.shape)
            elif i%100==0:
                print(i/self.data_num)
            sample=torch.reshape(sample,(784,)).to("cpu")
            X.append(sample)
            i+=1
            if i >= self.data_num:
                break
        print("finish building",i,"columns' X")
        
        X_matrix=torch.stack(X).to(dtype=torch.float32)
        print("full matrix is",X_matrix.shape)
        svd_X=X_matrix[label_idx,:]
        svd_X=torch.reshape(svd_X,[len(label_idx),784])
        print("svd_X is",svd_X.shape)
        del X_matrix
        self.aveX=torch.mean(svd_X,dim=0)
        print(self.aveX.shape)
        self.show_img(self.aveX)
        svd_X=svd_X-self.aveX
        self.X_loaded=True
        torch.save(svd_X,"/Volumes/DataHub/dataProcessed/X.pt")
        print("SVD start! X shape is ",svd_X.shape)

        U,S,Vh=torch.linalg.svd(svd_X.T,full_matrices=False)

        print("SVD Completed!","U shape is", U.shape)

        torch.save(U,"/Volumes/DataHub/dataProcessed/eigenMatrix.pt")

        print("U matrix Saved!")
        self.U=U
        self.get_selected_U()
    
    def load_eigenMatrix(self,U_dir="/Volumes/DataHub/dataProcessed/eigenMatrix.pt"):
        U=torch.load(U_dir)
        self.U=U.to("cpu")
        print("Loaded U",self.U.shape)
    
    def get_selected_U(self):
        self.Ur=self.U[:,:self.r]
        self.Ur=self.Ur.to("cpu")
        print("Build ",self.r,"columns Ur!",self.Ur.shape)
    
    def load_image(self,img_path):
        img=Image.open(img_path)
        img_tensor=self.common_trans(img)
        img_tensor=img_tensor.to("mps")
        img=torch.reshape(img_tensor,(784,)).to(device="cpu",dtype=torch.float32)
        return img
    
    def show_img(self,img_tensor):
        img_tensor=torch.reshape(img_tensor,(28,28))
        transform=v2.ToPILImage()
        img=transform(img_tensor)
        img.show()


    def img_rebuild(self,img_path):
        img=self.load_image(img_path)
        
        re_img=self.Ur @ (self.Ur.T @ img)+self.aveX 

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
    SVD=svd(r=60000,data_num=60000,label=5,dataset_dir="/Volumes/DataHub/dataProcessed/mnist/datasetSVD")
    SVD.show_img(SVD.aveX)
    SVD.img_rebuild("/Users/jingyi/Desktop/image.jpg")

