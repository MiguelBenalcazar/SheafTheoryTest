import torch
import os
import logging
from dataset_functions.datasetProcess import DatasetInformation
from models.models import RMNIST_CNN, SmallVGG, SimpleMLP, SimpleCNN



logger = logging.getLogger(__name__)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOAD DATASETS
    dataset_cifar10 = DatasetInformation(name="cifar10")
    # dataset_train_1 = dataset_cifar10.dataset_Training()
    dataset_test_1 = dataset_cifar10.dataset_Testing()

    dataset_r_mnist = DatasetInformation(name="r_mnist")
    # dataset_train_2 = dataset_r_mnist.dataset_Training()
    dataset_test_2 = dataset_r_mnist.dataset_Testing()


    data_loader_test_1 = torch.utils.data.DataLoader(
        dataset_test_1,
        batch_size=128,
        num_workers=10,
        drop_last=False,
    )

    data_loader_test_2 = torch.utils.data.DataLoader(
        dataset_test_2,
        batch_size=128,
        num_workers=10,
        drop_last=False,
    )

    model_1 = SimpleCNN(num_classes=dataset_cifar10.num_classes)
    model_1.to(device)

    model_2 = SimpleMLP(num_classes=dataset_r_mnist.num_classes)
    model_2.to(device)

    # Projection layers (latent_dim = 1024)
    latent_dim = 128

    P12 = torch.nn.Linear(sum(p.numel() for p in model_1.parameters()), latent_dim, bias=False)
    P21 = torch.nn.Linear(sum(p.numel() for p in model_2.parameters()), latent_dim, bias=False)
    P12.to(device)
    P21.to(device)



    # Load best Model 1 and P12
    ckpt1 = torch.load("best_model1.pth")
    model_1.load_state_dict(ckpt1["model"])
    P12.load_state_dict(ckpt1["P12"])

    # Load best Model 2 and P21
    ckpt2 = torch.load("best_model2.pth")
    model_2.load_state_dict(ckpt2["model"])
    P21.load_state_dict(ckpt2["P21"])


    model_1.eval()
    model_2.eval()

    with torch.no_grad():
            for i, (samples, labels) in enumerate(data_loader_test_1):
                samples = samples.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs_1 = model_1(samples)   
                loss_model_1 = loss_1(outputs_1, labels)
                test_loss_1 += loss_model_1.item()

                preds = outputs_1.argmax(dim=1)
                correct_1 += (preds == labels).sum().item()
                total_1 += labels.size(0)

            acc_1 = correct_1 / total_1
            logger.info(f"Model 1 - Loss: {test_loss_1/len(data_loader_test_1):.4f}, Acc: {acc_1:.4f}")


            for i, (samples, labels) in enumerate(data_loader_test_2):
                samples = samples.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)


                outputs_2 = model_2(samples)   
                loss_model_2 = loss_2(outputs_2, labels)
                test_loss_2 += loss_model_2.item()

                preds = outputs_2.argmax(dim=1)
                correct_2 += (preds == labels).sum().item()
                total_2 += labels.size(0)

