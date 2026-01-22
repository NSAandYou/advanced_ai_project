from abc import ABC, abstractmethod
from sklearn.tree import DecisionTreeClassifier

import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.tree import export_text
import numpy as np

class BaseModel():
    @abstractmethod
    def __init__(self, num_features, num_classes):
        pass

    @abstractmethod
    def train(self, X_train, y_train, batch_size):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class DTModel(BaseModel):
    def __init__(self, num_features, num_classes, seed=42):
        self.num_features = num_features
        self.model = DecisionTreeClassifier(random_state=seed)

    def train(self, X_train, y_train, batch_size=10):
        ## Remove time dimension
        if len(X_train.shape) > 2:
            X_train = X_train.reshape((-1, self.num_features))
            y_train = y_train.reshape((-1, 1))

        self.model.fit(X_train, y_train)
        ##print(export_text(self.model))

    def predict(self, X):
        input_shape = X.shape

        ## Remove time dimension
        if len(X.shape) > 2:
            X = X.reshape((-1, self.num_features))

        prediction = self.model.predict(X)

        ## Add time dimension
        if len(input_shape) > 2:
            prediction = prediction.reshape((input_shape[0], input_shape[1]))
        
        return prediction
        

class FFModel(BaseModel):
    class M(nn.Module):
        def __init__(self, num_features, num_classes):
            super(FFModel.M, self).__init__()
            self.cnn1 = nn.Conv1d(1, 3, kernel_size=9, stride=3)
            self.cnn2 = nn.Conv1d(3, 3, kernel_size=9, stride=3)
            self.cnn3 = nn.Conv1d(3, 1, kernel_size=1, stride=1)
            self.fc1 = nn.Linear(64, 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, num_classes)
            self.softmax = nn.Softmax()

        def forward(self, x):
            x = F.selu(self.cnn1(x))
            x = F.selu(self.cnn2(x))
            x = F.selu(self.cnn3(x))
            x = torch.flatten(x, start_dim=1)
            x = F.selu(self.fc1(x))
            x = F.selu(self.fc2(x))
            x = F.selu(self.fc3(x))
            return x

    def __init__(self, num_features, num_classes):
        self.model = FFModel.M(num_features, num_classes)
        self.crit = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.num_features = num_features

    def train(self, X_train, y_train, batch_size=10):
        ## Remove time dimension
        if len(X_train.shape) > 2:
            X_train = X_train.reshape((-1, 1, self.num_features))
            y_train = y_train.reshape((-1))

        for idx in range(0, X_train.shape[0], batch_size):
            data = torch.from_numpy(X_train[idx:min(idx+batch_size, X_train.shape[0])].astype(np.float32))
            label = torch.from_numpy(y_train[idx:min(idx+batch_size, y_train.shape[0])].astype(np.long))
            self.optim.zero_grad()
            pred = self.model(data)
            loss = self.crit(pred, label)
            loss.backward()
            self.optim.step()

    def predict(self, X):
        input_shape = X.shape

        ## Remove time dimension
        if len(X.shape) > 2:
            X = X.reshape((-1, 1, self.num_features))

        X = torch.from_numpy(X.astype(np.float32))

        prediction = self.model(X)
        prediction = prediction.detach().numpy()
        prediction = prediction.argmax(axis=1)

        ## Add time dimension
        if len(input_shape) > 2:
            prediction = prediction.reshape((input_shape[0], input_shape[1]))
        return prediction

class LSTMModel(BaseModel):
    class M(nn.Module):
        def __init__(self, num_features, num_classes):
            super(LSTMModel.M, self).__init__()
            self.fc1 = nn.Linear(num_features, 16)
            self.fc2 = nn.Linear(16, num_classes)
            self.lstm1 = nn.LSTM(16, 16, 2, batch_first=True)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = F.selu(self.fc1(x))
            x, (self.h0, self.c0) = self.lstm1(x, (self.h0.detach(), self.c0.detach()))
            x = F.selu(x)

            x = F.selu(self.fc2(x))
            x = self.softmax(x)
            return x
        
        def reset(self, num_layer=2, num_batches=1, num_states=16):
            self.h0 = torch.zeros(num_layer, num_batches, num_states) ## layer, batch, states
            self.h1 = torch.zeros(num_layer, num_batches, num_states)
            self.h2 = torch.zeros(num_layer, num_batches, num_states)
            self.h3 = torch.zeros(num_layer, num_batches, num_states)
            self.c0 = torch.zeros(num_layer, num_batches, num_states)
            self.c1 = torch.zeros(num_layer, num_batches, num_states)
            self.c2 = torch.zeros(num_layer, num_batches, num_states)
            self.c3 = torch.zeros(num_layer, num_batches, num_states)


    def __init__(self, num_features, num_classes):
        self.model = LSTMModel.M(num_features, num_classes)
        self.crit = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def train(self, X_train, y_train, batch_size=10):
        for idx_time in range(0, X_train.shape[0]):
            self.model.reset()
            for idx_samp in range(0, X_train.shape[1], batch_size):
                data = torch.from_numpy(X_train[idx_time,idx_samp:min(idx_samp+batch_size, X_train.shape[1])][np.newaxis,:,:].astype(np.float32))
                label = torch.from_numpy(y_train[idx_time,idx_samp:min(idx_samp+batch_size, y_train.shape[1])][np.newaxis,:].astype(np.long)).squeeze()
                self.optim.zero_grad()
                pred = self.model(data).squeeze()
                loss = self.crit(pred, label)
                loss.backward()
                self.optim.step()
            

    def predict(self, X):
        self.model.reset(num_batches=X.shape[0])
        X = torch.from_numpy(X.astype(np.float32))

        prediction = self.model(X)
        prediction = prediction.detach().numpy()
        prediction = prediction.argmax(axis=2)

        return prediction


##if "__main__" == __name__:
##    from torchsummary import summary
##    summary(LSTMModel.M(600, 15),(1,600))