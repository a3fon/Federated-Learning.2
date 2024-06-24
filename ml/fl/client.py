import torch

from typing import Dict, Tuple, List, Union, Optional, Any
from collections import OrderedDict
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from ml.utils.train_utils import train, test

import numpy as np

class Client:
    def __init__(self, id, dataset):
        self.id = id
        self.dataset = dataset
        self.trainloader = None
        self.testloader = None
        self.model = None
        self.optimizer = None
        self.epochs = None
        self.lr = None
        self.criterion = None
        self.device = None
        self.test_size = None
        self.batch_size = None


    def init_parameters(self, params: Dict[str, Union[bool, str, int, float]], model):  # default parameters
        self.epochs = params["epochs"]
        self.lr = params["lr"]
        self.model = model
        self.device = params["device"]
        self.test_size = params['test_size']
        self.batch_size = params['batch_size']

        # Get Criterion
        from ml.utils.helpers import get_criterion
        self.criterion = get_criterion(params['criterion'])

        # Get Optimizer
        from ml.utils.helpers import get_optim
        self.optimizer = get_optim(model, params['optimizer'], self.lr)


        # Train - Test Split
        train_set, val_set = random_split(self.dataset, [int(len(self.dataset)*(1 - self.test_size)), int(len(self.dataset)*self.test_size)])

        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=True)



    def set_parameters(self, parameters: Union[List[np.ndarray], torch.nn.Module]):
        if not isinstance(parameters, torch.nn.Module):
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
        else:
            self.model.load_state_dict(parameters.state_dict(), strict=True)

    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]


    def update(self):
        train_history = train(self.model,self.train_loader, self.device, self.criterion, self.optimizer, self.epochs,False)

    def evaluate(self, test_loader):
        acc, f1 = test(self.model,test_loader,self.criterion, self.device)
        return acc, f1
# import torch
# from torch.utils.data import random_split, DataLoader
# from typing import Dict, Union, List
# import numpy as np
# from collections import OrderedDict
# from ml.utils.train_utils import train, test
#
# class Client:
#     def __init__(self, id, dataset):
#         self.id = id
#         self.dataset = dataset
#         print(self.dataset)
#         self.train_loader = None
#         self.test_loader = None
#         self.model = None
#         self.optimizer = None
#         self.epochs = None
#         self.lr = None
#         self.criterion = None
#         self.device = None
#         self.test_size = None
#         self.batch_size = None
#
#     def adjust_splits(self, dataset, test_size):
#         """ Adjust dataset split sizes to avoid rounding issues with random_split. """
#         total_size = len(dataset)
#         print(dataset)
#         train_size = int(total_size * (1 - test_size))
#         print(train_size)
#         test_size = total_size - train_size
#         print(test_size)
#         return random_split(dataset, [train_size, test_size])
#
#     def init_parameters(self, params: Dict[str, Union[bool, str, int, float]], model):
#         self.epochs = params["epochs"]
#         self.lr = params["lr"]
#         self.model = model.to(params["device"])
#         self.device = params["device"]
#         self.test_size = params['test_size']
#         self.batch_size = params['batch_size']
#
#         # Get Criterion
#         from ml.utils.helpers import get_criterion
#         self.criterion = get_criterion(params['criterion'])
#
#         # Get Optimizer
#         from ml.utils.helpers import get_optim
#         self.optimizer = get_optim(model, params['optimizer'], self.lr)
#
#         # Train - Test Split using the adjusted split function
#         train_set, val_set = self.adjust_splits(self.dataset, self.test_size)
#
#         self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
#         self.test_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=True)
#
#     def set_parameters(self, parameters: Union[List[np.ndarray], torch.nn.Module]):
#         if not isinstance(parameters, torch.nn.Module):
#             params_dict = zip(self.model.state_dict().keys(), parameters)
#             state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
#             self.model.load_state_dict(state_dict, strict=True)
#         else:
#             self.model.load_state_dict(parameters.state_dict(), strict=True)
#
#     def get_parameters(self) -> List[np.ndarray]:
#         return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
#
#     def update(self):
#         train_history = train(self.model, self.train_loader, self.device, self.criterion, self.optimizer, self.epochs, False)
#
#     def evaluate(self, test_loader):
#         acc, f1 = test(self.model, test_loader, self.criterion, self.device)
#         return acc, f1
