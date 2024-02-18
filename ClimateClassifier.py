#!/usr/bin/env python
# coding: utf-8
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import random
import torchvision
import asyncio
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from PIL import Image
from io import BytesIO
from pathlib import Path
app = Flask(__name__)
CORS(app)

def predict_external_image(image_name):
    image_bytes = BytesIO()
    image_name = image_name.convert("RGB")
    image_name.save(image_bytes, format='JPEG')
    image_bytes.seek(0)
    image = Image.open(image_bytes)
    example_image = transformations(image)
   # plt.imshow(example_image.permute(1, 2, 0))
   # plt.show()
    return {"message": "The image resembles " + predict_image(example_image, model) + "."}

# evaluated model already
@app.route('/', methods=['GET', 'POST'])
def index():
    return 'Hello, World!'
# API endpoint for predicting an image
@app.route('/predict', methods=['GET','POST'])
def predict():
    image_file = request.files['image']
    image = Image.open(image_file)
    message = predict_external_image(image)
    return jsonify({'message': message})

def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]

#predict_external_image('/Users/prashantkondayapalepu/Downloads/IMG_1173.jpg')



# In[ ]:





if __name__ == '__main__':


    # In[2]:



    #Load the pre trained model





    data_dir  = '/Users/prashantkondayapalepu/Downloads/AirPollution/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset/IND_and_NEP'

    classes = os.listdir(data_dir)
    #print(classes)


    # In[ ]:



    transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    dataset = ImageFolder(data_dir, transform = transformations)

    #dataset


    # In[ ]:



    #get_ipython().run_line_magic('matplotlib', 'inline')

    def show_sample(img, label):
        print("Label:", dataset.classes[label], "(Class No: "+ str(label) + ")")
        plt.imshow(img.permute(1, 2, 0))
        plt.show()


    # In[ ]:


    img, label = dataset[12]
    #show_sample(img, label)


    # In[ ]:


    img, label = dataset[7]
    #show_sample(img, label)


    # In[ ]:


    img, label = dataset[2]
    #show_sample(img, label)


    # In[ ]:


    random_seed = 42
    torch.manual_seed(random_seed)
    train_ds, val_ds, test_ds = random_split(dataset, [7716, 852, 3672])
    len(train_ds), len(val_ds), len(test_ds)

    # In[ ]:



    batch_size = 32


    # In[ ]:


    train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers = 4, pin_memory = True)


    # In[ ]:




    def show_batch(dl):
        for images, labels in dl:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(images, nrow = 16).permute(1, 2, 0))
            break


    # In[ ]:


    #show_batch(train_dl)


    # In[ ]:


    #show_batch(val_dl)


    # In[ ]:


    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    class ImageClassificationBase(nn.Module):

        def training_step(self, batch):
            images, labels = batch
            out = self(images)                  # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
            return loss

        def validation_step(self, batch):
            images, labels = batch
            out = self(images)                    # Generate predictions
            loss = F.cross_entropy(out, labels)   # Calculate loss
            acc = accuracy(out, labels)           # Calculate accuracy
            return {'val_loss': loss.detach(), 'val_acc': acc}

        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

        def epoch_end(self, epoch, result):
            print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))


    # In[ ]:


    class ResNet(ImageClassificationBase):
        def __init__(self):
            super().__init__()

            # Use a pretrained model
            self.network = models.resnet50(pretrained=True)

            # Replace last layer
            num_ftrs = self.network.fc.in_features
            self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))

        def forward(self, xb):
            return torch.sigmoid(self.network(xb))

    model = ResNet()


    # In[ ]:


    def get_default_device():

        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def to_device(data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list,tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl:
                yield to_device(b, self.device)

        def __len__(self):
            """Number of batches"""
            return len(self.dl)


    # In[ ]:


    device = get_default_device()
    device


    # In[ ]:


    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(model, device)


    # In[ ]:


    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)


    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)


    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        history = []
        optimizer = opt_func(model.parameters(), lr)
        counter = 0
        for epoch in range(epochs):
            # Training Phase
            model.train()
            train_losses = []
            for batch in train_loader:
                counter+=1
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if counter == 40:
                    break
            # Validation phase
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)
            print('hello')
        return history


    # In[ ]:


    model = to_device(ResNet(), device)


    # In[ ]:


    evaluate(model, val_dl)

    num_epochs = 1
    opt_func = torch.optim.Adam
    lr = 0.01

    #history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)





    # In[ ]:

    """
    img, label = test_ds[17]
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model), '\nMake sure to Recycle into Blue Recycle Bin!')
    
    
    # In[ ]:
    
    
    img, label = test_ds[30]
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model), '\nMake sure to Recycle into Blue Recycle Bin!')
    
    
    # In[ ]:
    
    
    img, label = test_ds[51]
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model), '\nMake sure to Recycle into Blue Recycle Bin!')
    
    
    # In[ ]:
    
    
    img, label = test_ds[21]
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model), '\nMake sure to Recycle into Blue Recycle Bin!')
    """

    # In[32]:






    app.run()

