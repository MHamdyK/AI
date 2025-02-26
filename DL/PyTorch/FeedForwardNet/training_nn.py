import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def load_MNIST_dataset():
    train_data = datasets.MNIST(
        root="data",
        train = True,
        transform = ToTensor(),
        download = True 
    )
    test_data = datasets.MNIST(
        root ="data",
        download = True,
        train = False,
        transform = ToTensor()
    )
    return train_data,test_data

class ModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=28*28,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features= 10)
        )
        self.softmax = nn.Softmax(dim=1)
        """
        torch.tensor([   C1  C2   C3 
                     S1:[0.1,0.8,0.2],
                     S2:[0.3,0.6,0.1],
                     S3:[0.7,0.2,0.1]
                     ])
        """
    def forward(self,input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions

def train_one_epoch(model,data_loader,device,optimizer,loss_fn):
    model.train()
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device),targets.to(device)
        #Calculate the loss
        predictions = model(inputs)
        loss = loss_fn(predictions,targets)

        #backpropagate to calculate the gradient descent and update it
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Loss: {loss.item()}")

def train(model,data_loader,device,optimizer,loss_fn,epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        train_one_epoch(model,data_loader,device,optimizer,loss_fn)    
        print("---------------------------------------------------")
    print("Training completed.")

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
if __name__ == "__main__":

    print("Downloading/loading MNIST dataset...")
    train_data,test_data = load_MNIST_dataset()
    print("MNIST dataset downloaded successfully")

    train_dataloader = DataLoader(dataset=train_data,batch_size = BATCH_SIZE,shuffle=True)

    #build the model
    # if torch.cude.is_available():
    #     device = "cuda"
    # else:
    #     device = "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using:{device} device")

    model = ModelV1().to(device)
    #optimizer and loss function
    optimizer = torch.optim.Adam(params=model.parameters(),lr= LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    train(model,train_dataloader,device,optimizer,loss_fn,epochs=EPOCHS)

    torch.save(model.state_dict(),"feedforwardnet.pth")
    
    print("Model trained and stored at feedforwardnet.pth")


