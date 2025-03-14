import torch
import torch.optim as optim
from torch.nn.modules import activation
from torchvision.transforms import v2
from datasets import load_from_disk
from torch import nn
from torch.utils.data import DataLoader, Dataset
class cnn(nn.Module):
    def __init__(self,kernel_size,dims,hidden_layers,dataset_dir,label_num):
        super().__init__()
        self.hidden_layers=hidden_layers
        self.kernal_size=kernel_size
        print("cnn initialized!")
        self.label_num=label_num
        self.dims=dims

    def build_nn(self):
        self._cnn=nn.Sequential(
            nn.Conv2d(1,16,kernel_size=self.kernal_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,kernel_size=self.kernal_size),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        layers=[]
        layers.append(nn.Flatten())
        for i in range(self.hidden_layers):
            layer=nn.Linear(self.dims[i],self.dims[i+1])
            activation=nn.ELU()
            layers.append(layer)
            layers.append(activation)
        layers.append(nn.Linear(self.dims[-1],self.label_num))
        layers.append(nn.Softmax(1))

        self._layers=nn.Sequential(*layers)

    def forward(self,x):
        x=self._cnn(x)
        x=self._layers(x)
        
        return x

class CustomDataset(Dataset):
    def __init__(self, hf_data, transform=None):
        self.data = hf_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dtype=v2.Compose(
            [v2.ToImage(),
             v2.ToDtype(torch.float32)]
        )
        image = self.data["image"][idx]
        image=dtype(image)
        label = self.data["label"][idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 0.001
    KERNEL_SIZE = 3
    DIMS = [800,128] 
    
    
    print("loading datasets")
    trainset_dir = "/Volumes/DataHub/dataProcessed/mnist/datasetCNN"
    testset_dir ="/Volumes/DataHub/dataProcessed/mnist/test/datasetCNN"
    train_data = load_from_disk(trainset_dir)
    test_data=load_from_disk(testset_dir)
    

    train_set = CustomDataset(train_data)
    test_set = CustomDataset(test_data)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    print("finish loading datasets!")
    model = cnn(
        kernel_size=KERNEL_SIZE,
        dims=DIMS,
        hidden_layers=len(DIMS)-1,
        dataset_dir=trainset_dir,
        label_num=train_data.features["label"].num_classes
    )
    
    model.build_nn()
    

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.to(device)
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("Start Trainning!")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} Acc: {train_acc:.2f}%")
    

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    print(f"Test Accuracy: {100 * test_correct / test_total:.2f}%")

if __name__ == "__main__":
    main()



