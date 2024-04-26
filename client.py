# client.py
import flwr as fl
import torch
import torch.nn as nn
from torch.optim import SGD
import torchvision.transforms as transforms
import torchxrayvision as xrv
from torchvision.models import resnet18
from collections import OrderedDict
# import PIL.Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Lambda
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np
# from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import *

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Load the pre-trained DenseNet model from torchxrayvision
        self.model = xrv.models.DenseNet(weights="densenet121-res224-nih")

        # Adjust the output features to 14
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, 14)

        # Check if model has op_threshs and adjust them if necessary
        if hasattr(self.model, 'op_threshs'):
            self.model.op_threshs = torch.randn(14)  # Adjust the thresholds to match 14 classes

    def forward(self, x):
        # Apply the model
        x = self.model(x)
        # If the model uses `op_norm`, modify or bypass it here
        return x

# # Load the dataset using torchxrayvision
# def load_data(datapath, csvpath):
#     def transform(x):
#         # Ensure the data type is uint8 and handle single channel data
#         if x.ndim == 3 and x.shape[0] == 1:  # Assuming (1, H, W) format for single-channel
#             x = x.squeeze(0)  # Reduce to (H, W) if single-channel grayscale
#         x = np.clip(x, a_min=0, a_max=255)  # Ensure the range is valid for uint8
#         x = x.astype(np.uint8)  # Convert type to uint8
#         return Image.fromarray(x)  # Convert to PIL Image
    
#     transform_pipeline = Compose([
#         transform,
#         ToTensor(),
#         Normalize(mean=[0.485], std=[0.229])  # Adjust these values based on your dataset specifics
#     ])
#         # Define a wrapper to apply transformation to image data
#     # Define a custom dataset wrapper to apply the transformation
#     class CustomDataset(torch.utils.data.Dataset):
#         def __init__(self, dataset, transform=None):
#             self.dataset = dataset
#             self.transform = transform

#         def __len__(self):
#             return len(self.dataset)

#         def __getitem__(self, idx):
#             data = self.dataset[idx]
#             image, label = data["img"], data["lab"]
#             if self.transform:
#                 image = self.transform(image)
#             return image, label

#     # Load the NIH Dataset with transformations
#     # dataset = xrv.datasets.NIH_Dataset(imgpath="data/images/", transform=transform)
#     # dataset = xrv.datasets.NIH_Dataset(imgpath="data/images/")


#     original_dataset = xrv.datasets.NIH_Dataset(imgpath=datapath, csvpath=csvpath)
#     dataset = CustomDataset(original_dataset, transform=transform_pipeline)


#     # Create the DataLoader
#     loader = DataLoader(dataset, batch_size=8, shuffle=True)
#     return loader

# Load the dataset using torchxrayvision
def load_data(datapath, pathtocsv):
# def load_test_data(datapath):
    def transform(x):
        # Ensure the data type is uint8 and handle single channel data
        if x.ndim == 3 and x.shape[0] == 1:  # Assuming (1, H, W) format for single-channel
            x = x.squeeze(0)  # Reduce to (H, W) if single-channel grayscale
        x = np.clip(x, a_min=0, a_max=255)  # Ensure the range is valid for uint8
        x = x.astype(np.uint8)  # Convert type to uint8
        return Image.fromarray(x)  # Convert to PIL Image
    
    transform_pipeline = Compose([
        transform,
        ToTensor(),
        Normalize(mean=[0.485], std=[0.229])  # Adjust these values based on your dataset specifics
    ])
        # Define a wrapper to apply transformation to image data
    # Define a custom dataset wrapper to apply the transformation
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            data = self.dataset[idx]
            image, label = data["img"], data["lab"]
            if self.transform:
                image = self.transform(image)
            return image, label

    # Load the NIH Dataset with transformations
    # dataset = xrv.datasets.NIH_Dataset(imgpath="data/images/", transform=transform)
    # dataset = xrv.datasets.NIH_Dataset(imgpath="data/images/")


    # original_dataset = xrv.datasets.NIH_Dataset(imgpath=datapath, csvpath=csvpath)
    # datapath = ""
    # datapath = "./"
    # pathtocsv = os.path.join(datapath, "test_data.csv")
    # pathtocsv = os.path.join(datapath, "Data_Entry_2017_v2020.csv.gz")
    original_dataset = xrv.datasets.NIH_Dataset(imgpath=datapath, csvpath=pathtocsv)
    dataset = CustomDataset(original_dataset, transform=transform_pipeline)


    # Create the DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return loader



# Define the training logic
def train(model, trainloader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

# def evaluate_model(model, testloader):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()
#     accuracy, precision, recall, f1 = 0, 0, 0, 0
#     with torch.no_grad():
#         for images, labels in testloader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(torch.sigmoid(outputs), 1)
#             accuracy += accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
#             precision += precision_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
#             recall += recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
#             f1 += f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')

#     accuracy /= len(testloader)
#     precision /= len(testloader)
#     recall /= len(testloader)
#     f1 /= len(testloader)
#     return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# def evaluate_model(model, testloader):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()
#     accuracy, precision, recall, f1 = 0, 0, 0, 0
#     with torch.no_grad():
#         for images, labels in testloader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(torch.sigmoid(outputs), 1)
#             accuracy += balanced_accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
#             # precision += precision_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
#             # recall += recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
#             # f1 += f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')

#     accuracy /= len(testloader)
#     # precision /= len(testloader)
#     # recall /= len(testloader)
#     # f1 /= len(testloader)
#     return {"accuracy": accuracy}#, "precision": precision, "recall": recall, "f1": f1}


def evaluate_model(model, testloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    loss, accuracy, precision, recall, f1 = 0.0, 0, 0, 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # loss = criterion(outputs, labels)
            criterion = nn.BCEWithLogitsLoss()
            loss += criterion(outputs, labels).item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()  # Applying threshold
            accuracy += accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
            precision += precision_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='samples')
            recall += recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='samples')
            f1 += f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='samples')

    accuracy /= len(testloader)
    precision /= len(testloader)
    recall /= len(testloader)
    f1 /= len(testloader)

    print({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})

    return loss, accuracy
    # return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Flower client
class XrayClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = SimpleCNN()
        self.trainloader = load_data(datapath="data/train/", pathtocsv= "data/train_data.csv")
        self.testloader  = load_data(datapath="data/test/",  pathtocsv= "data/test_data.csv")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model = train(self.model, self.trainloader, epochs=1)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    # def evaluate(self, parameters, config):
    #     self.set_parameters(parameters)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.model.to(device)
    #     self.model.eval()
    #     criterion = nn.CrossEntropyLoss()
    #     test_loss, correct = 0, 0
    #     with torch.no_grad():
    #         # for images, labels in self.trainloader:
    #         for images, labels in self.testloader:
    #             images, labels = images.to(device), labels.to(device)
    #             outputs = self.model(images)
    #             test_loss += criterion(outputs, labels).item()
    #             correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

    #     accuracy = correct / len(self.trainloader.dataset)
    #     return float(test_loss / len(self.trainloader)), len(self.trainloader.dataset), {"accuracy": accuracy}

    # def evaluate(self, parameters, config):
    #     self.set_parameters(parameters)
    #     metrics = evaluate_model(self.model, self.testloader)
    #     return metrics

    def evaluate(self, parameters, config):
        # Set the model parameters (weights)
        self.set_parameters(parameters)

        # Perform evaluation using the existing evaluate_model function
        # metrics = evaluate_model(self.model, self.testloader)

        # # Extract loss if available or assume loss is not calculated by evaluate_model
        # loss = metrics.get("loss", 0.0)  # Default loss to 0 if not calculated

        # # Calculate the total number of examples evaluated
        num_examples = len(self.testloader.dataset)

        # # Remove loss from metrics if it's there, as we need to return it separately
        # if "loss" in metrics:
        #     del metrics["loss"]

        # # The method must return a tuple with (loss, num_examples, metrics_dict)
        # return loss, num_examples, {"f1": metrics['f1']}

        loss, accuracy = evaluate_model(self.model, self.testloader)
        # return float(loss), num_examples["testset"], {"accuracy": float(accuracy)} --wrong
        return float(loss), num_examples, {"accuracy": float(accuracy)}




if __name__ == "__main__":
    fl.client.start_client(server_address="0.0.0.0:8080", client=XrayClient().to_client())
