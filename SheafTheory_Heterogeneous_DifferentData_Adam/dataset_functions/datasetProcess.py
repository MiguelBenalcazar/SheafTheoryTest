from torchvision import datasets, transforms

class DatasetInformation:
    def __init__(self, name):
        self.name = name
        if name == "cifar10":
            self.num_classes = 10
            self.input_size = (3, 32, 32)
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2023, 0.1994, 0.2010)
            self.transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(self.mean, self.std)
                ])
            
        elif name == "r_mnist":
            self.num_classes = 10
            self.input_size = (1, 28, 28)
            self.mean = (0.1307,)
            self.std = (0.3081,)
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=360),  # rotate between 0–360 degrees
                transforms.ToTensor(), 
                transforms.Normalize(self.mean, self.std)  # standard MNIST normalization
                ])
        else:
            raise ValueError(f"Unknown dataset: {name}")
        
    def dataset_Training(self):
        if self.name == "cifar10":
            return datasets.CIFAR10(
                root="./data",
                train=True,
                download=True,
                transform=self.transform
            )
        elif self.name == "r_mnist":
            return datasets.MNIST(
                root="./data",
                train=True,
                download=True,
                transform=self.transform
            )
        
    def dataset_Testing(self):
        if self.name == "cifar10":
            return datasets.CIFAR10(
                root="./data",
                train=False,
                download=True,
                transform=self.transform
            )
        elif self.name == "r_mnist":
            return datasets.MNIST(
                root="./data",
                train=False,
                download=True,
                transform=self.transform
            )