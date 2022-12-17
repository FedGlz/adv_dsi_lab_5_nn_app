import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim.lr_scheduler as op
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

class PytorchMultiClass(nn.Module):
    def __init__(self, num_features):
        super(PytorchMultiClass, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 34) 
        self.layer_out = nn.Linear(34, 105)
    
    
    def forward(self,x):
        x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
        x = self.layer_out(x)
        return(x)
    
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0') 
    else:
        device = torch.device('cpu')
    return device

class PytorchDataset(Dataset):
    """
    Pytorch dataset
    
    Atributes
    -------------
    X_tensor: Pytorch Tensor
    Y_tensor: Pytorch Tensor
        Target Tensor
    
    Methods
    -------------
    __getitem__(index)
        Returns features and target for a given index
    __len__ 
        Returns the number of observations
    to_tensor(data)
        Converts pandas series to pytorch tensor
    """
    def __init__(self, X, y):
        self.X_tensor = self.to_tensor(X)
        self.y_tensor = self.to_tensor(y)
    
    def __getitem__(self,index):
        return self.X_tensor[index], self.y_tensor[index]
    
    def __len__(self):
        return len(self.X_tensor)
    
    def to_tensor(self, data):
        return torch.Tensor(np.array(data))

    
def train_classification (train_data, model, criterion, optimizer, batch_size, device, scheduler=None):
    
    """ Train a Pytorch multi-class classification problem
    
    Parameters
    -----------------------
    train_data : torch.utils.data.Dataset
        Pytorch Dataset
    model      : torch nn.Module
        Pytorch model
    criterion  : function
            Loss function
    optimizer  : torch.optimizer
            Optimizer
    batch_size : int 
            No of observations per batch
    device     : str
            Name of the device used by model
    scheduler  : torch.optim.lr_scheduler
            Pytorch scheduler used for updating learning rate
    collate_fn : function
            Function defining required pre-processing steps
    
    Returns
    -----------------
    Float: Loss Score
    Float: Accuracy Score
    
    """
    
    #Set model to training mode
    model.train()
    train_loss = 0
    train_acc = 0
    
    #Create data loader
    data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    #Iterate through data by batch of observations
    for feature, target_class in data:
        
        #Reset gradients
        optimizer.zero_grad()
        
        #Load data to specified device
        feature, target_class = feature.to(device), target_class.to(device)
        
        #Make predictions
        output = model(feature)
        
        #Calculate loss for a given batch
        loss = criterion(output, target_class.long())
        
        #Calculate global loss
        train_loss += loss.item()
        
        #Calculate gradients
        loss.backward()
        
        #Update Weights
        optimizer.step()
        
        #Calculate Global Accuracy
        train_acc += (output.argmax(1) == target_class).sum().item()
    
    #Adust Learning Rate
        
    if scheduler:
        scheduler.step()
    
    return train_loss / len(train_data) , train_acc / len(train_data)
        
    
def test_classification (test_data, model, criterion, batch_size, device):
    """ Train a Pytorch multi-class classification problem
    
    Parameters
    -----------------------
    test_data : torch.utils.data.Dataset
        Pytorch Dataset
    model      : torch nn.Module
        Pytorch model
    criterion  : function
            Loss function
    batch_size : int 
            No of observations per batch
    device     : str
            Name of the device used by model
    collate_fn : function
            Function defining required pre-processing steps
    
    Returns
    -----------------
    Float: Loss Score
    Float: Accuracy Score
    
    """
    #Set model to training mode
    model.eval()
    test_loss=0
    test_acc=0
    
    #Create data loader
    data = DataLoader(test_data, batch_size=batch_size)
    
    #Iterate through data by batch of observations
    for feature, target_class in data:
        
        #Load data to specified device
        feature, target_class = feature.to(device), target_class.to(device)
        
        #Set no update to gradients
        with torch.no_grad():
            
            #Make predictions
            output = model(feature)
            
            #Calculate loss for a given batch
            loss = criterion(output, target_class.long())
            
            #Calculate global loss
            test_loss += loss.item()

            #Calculate Global Accuracy
            test_acc += (output.argmax(1) == target_class).sum().item()
    
    return test_loss/ len(test_data) , test_acc / len(test_data)
    

    
class MultiColumnOrdinalEncoder:
    def __init__(self,columns = None):
        self.columns = columns 

    def fit(self,X,y=None):
        return self 

    def transform(self,X):
        output = X.copy()
        
        if self.columns is not None:
            for col in self.columns:
                encoder = ce.OrdinalEncoder(cols= col)
                output[col] = encoder.fit_transform(output[col])
        
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns 

    def fit(self,X,y=None):
        return self 

    def transform(self,X):
        output = X.copy()
        
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)    

    
