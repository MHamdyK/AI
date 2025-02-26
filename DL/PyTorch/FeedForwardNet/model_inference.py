from training_nn import ModelV1,load_MNIST_dataset
import torch
from torch import nn
from torch.utils.data import DataLoader

target_mapping= ["one","two","Three","Four","Five","Six","Seven","Eight","Nine","Ten"]
BATCH_SIZE = 10
def validation(model,data_loader,loss_fn,device):
    model.eval()
    with torch.inference_mode():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            #calculate the loss
            predictions = model(inputs)
            loss = loss_fn(predictions,targets)
            print(f"Loss:{loss}")
            predicted = predictions[0].argmax(dim=0)
            print(f"Predicted:{target_mapping[predicted]} | Ground Truth:{target_mapping[targets.argmax(dim=0)]}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    uploaded_model = ModelV1()
    state_dict = torch.load("/media/mohamdy/HDD 1/_repos/AI/DL/PyTorch/FeedForwardNet/feedforwardnet.pth")
    uploaded_model.load_state_dict(state_dict)
    uploaded_model.to(device)

    #load the MNIST dataset the validation version
    _,test_data = load_MNIST_dataset()
    test_data_loader = DataLoader(dataset = test_data, batch_size=BATCH_SIZE,shuffle= False)
    loss_fn = nn.CrossEntropyLoss()
    validation(uploaded_model,test_data_loader,loss_fn,device)