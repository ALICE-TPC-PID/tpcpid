from copy import copy
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn import preprocessing
from sklearn.pipeline import Pipeline

import timeit
import numpy as np


class dataset(Dataset):

    def __init__(self, X, y):

        self.list = list(zip(X, y))
        self.element_size_X = X.element_size()
        self.element_size_y = y.element_size()
        self.length = len(self.list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.list[idx]
    
    def mem_size(self):
        return self.element_size_X*self.length + self.element_size_y*self.length

# This class actually loads the data and saves it in dataset objects


class DataLoading(dataset):

    def __init__(self, training_data, validation_data, batch_sizes=[1], num_workers=0,
                 X_data_scalers=[
                     ('box-cox', preprocessing.PowerTransformer(method='box-cox', standardize=True))],
                 y_data_scalers=[
                     ('standard scaler', preprocessing.StandardScaler())],
                 transform_data=True, shuffle_every_epoch=True):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = num_workers

        self.transform_data = transform_data

        self.loadTS = False
        self.loadVS = False

        self.shuffle_every_epoch = shuffle_every_epoch

        print("\n =============== Data preparation =============== \n")

        start_time_loader = timeit.default_timer()

        if ((not self.loadTS) and (not self.loadVS)):

            self.batch_sizes = batch_sizes

            if not self.loadTS:
                if not transform_data:
                    self.transformers_X = []
                    self.transformers_Y = []
                    print("No transformation is performed on training data!")
                else:
                    self.transformers_X = X_data_scalers
                    self.transformers_Y = y_data_scalers
                    print("Transforming training data...")

                self.scalingX = ScalingX(scalers = self.transformers_X)
                self.scalingY = ScalingY(scalers = self.transformers_Y)

                self.datasetTS = dataset(self.scalingX.scale(training_data[0]).float().to(self.device),
                                         self.scalingY.scale(training_data[1]).float().to(self.device))
                self.inverse_X = InverseScaling(self.scalingX.fitted_scalers_X)
                self.inverse_Y = InverseScaling(self.scalingY.pipe_y)

                self.sizeTS = self.datasetTS.mem_size()
                self.loadTS = True
                print("Training data transformed.")
                print("\nContinuing with validation data...\n")

            if not self.loadVS:

                if not transform_data:
                    print("\nNo transformation is performed on validation data!")
                else:
                    print("\nTransforming validation data...")
                self.datasetVS = dataset(self.scalingX.scale(validation_data[0]).float().to(self.device),
                                         self.scalingY.scale(validation_data[1]).float().to(self.device))

                self.sizeVS = self.datasetVS.mem_size()
                self.loadVS = True

                print("Validation data transformed.\n")

        end_time_loader = timeit.default_timer()

        print("Duration:", np.round(end_time_loader-start_time_loader, 3), "s")
        print("Training data:", len(self.datasetTS), "elements - Memory size: ", self.sizeTS/1e6, "MiB\n",
              "Validation data:", len(self.datasetVS), "elements - Memory size: ", self.sizeVS/1e6, "MiB\n")
        print("Data is loaded, Training can begin!\n")
        print("================================================\n")

class ScalingX:
    
    def __init__(self, scalers=[], newscale=True, copy_to_dev=True):
        self.scalers = scalers
        self.newscale = newscale
        self.copy_to_dev = copy_to_dev
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def scale(self, data):

        if self.newscale:
            if not self.scalers:
                self.pipe_X = False
                self.fitted_scalers_X = False
                transformed_data = torch.tensor(data)
            else:
                self.pipe_X = Pipeline(self.scalers)
                self.fitted_scalers_X = self.pipe_X.fit(data)
                transformed_data = torch.tensor(self.fitted_scalers_X.transform(data))

        else:
            if self.fitted_scalers_X:
                transformed_data = torch.tensor(self.fitted_scalers_X.transform(data))
            else:
                transformed_data = torch.tensor(data)

        if self.copy_to_dev:
            transformed_data = transformed_data.to(self.device)
        
        self.newscale=False

        return transformed_data.float()


class ScalingY:
    
    def __init__(self, scalers=[], newscale=True, copy_to_dev=True):
        self.scalers = scalers
        self.newscale = newscale
        self.copy_to_dev = copy_to_dev
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def scale(self, data):

        if self.newscale:
            if not self.scalers:
                self.pipe_y = False
                self.fitted_scalers_y = False
                transformed_data = torch.tensor(data)
            else:
                self.pipe_y = Pipeline(self.scalers)
                self.fitted_scalers_y = self.pipe_y.fit(data)
                transformed_data = torch.tensor(self.fitted_scalers_y.transform(data))

        else:
            if self.fitted_scalers_y:
                transformed_data = torch.tensor(
                    self.fitted_scalers_y.transform(data))
            else:
                transformed_data = torch.tensor(data)

        if self.copy_to_dev:
            transformed_data = transformed_data.to(self.device)
        
        self.newscale=False

        return transformed_data.float()


class InverseScaling:
    
    def __init__(self, scalers_fit, copy_to_dev=True):
        self.scalers_fit = scalers_fit
        self.copy_to_dev = copy_to_dev
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def scale(self, data):
        output = torch.tensor(self.scalers_fit.inverse_transform(data))

        if self.copy_to_dev:
            output = output.to(self.device)

        return output.float()
