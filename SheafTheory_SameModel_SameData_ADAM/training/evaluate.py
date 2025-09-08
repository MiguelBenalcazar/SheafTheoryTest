import logging
import torch
import torch.nn.functional as F
logger = logging.getLogger(__name__)


def evaluate_models(
        loss_1,
        loss_2,
        model_1, 
        model_2, 
        P12,
        P21,
        data_loader_test_1,
        data_loader_test_2,
        writer,
        epoch, 
        device, write_data = True
        ):
    
    logger.info(f"-------VALIDATION Epoch: {epoch} -------")

    model_1.eval()
    model_2.eval()

    correct_1, total_1, test_loss_1 = 0, 0, 0.0
    correct_2, total_2, test_loss_2= 0, 0, 0.0

    # EXTRACT DATA FROM MODELS 
    
    model_1_projection_list = []
    model_2_projection_list = []

    def hook_model1(module, input, output):
        # output here is fc1(x) (pre-ReLU), so apply ReLU manually
        flattened = output.view(output.size(0), -1)
        model_1_projection_list.append(flattened.detach().cpu())
            

    def hook_model2(module, input, output):
            # output here is fc1(x) (pre-ReLU), so apply ReLU manually
        flattened = output.view(output.size(0), -1)
        model_2_projection_list.append(flattened.detach().cpu())

    # Register the hook
    hook1 = model_1.conv_block3.register_forward_hook(hook_model1)
    hook2 = model_2.conv_block3.register_forward_hook(hook_model2)
    
    

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

        acc_2 = correct_2 / total_2
        logger.info(f"Model 2 - Loss: {test_loss_2/len(data_loader_test_2):.4f}, Acc: {acc_2:.4f}")


        # Remove hooks after evaluation
        hook1.remove()
        hook2.remove()

        # ---- Verify discrepancy between restriction maps ----
        # Concatenate all batches: [num_samples, fc1_dim]
        projection_all_model_1 = torch.cat(model_1_projection_list, dim=0).to(device) 
        projection_all_model_2 = torch.cat(model_2_projection_list, dim=0).to(device)  

        # Project with P12
        proj_all_model1_P12 = P12(projection_all_model_1).detach().cpu()
        # Project with P21
        proj_all_model2_P21 = P21(projection_all_model_2).detach().cpu()
        discrepancy = proj_all_model1_P12 - proj_all_model2_P21

        normalized_discrepancy = torch.sum(discrepancy**2) / (projection_all_model_1.size(0) * discrepancy.size(1))

        if write_data:
          
            logger.info(f"Restriction map discrepancy: {discrepancy.mean().item()} Discrepancy Similarity: {  torch.sum(discrepancy**2).item()}, normalized_discrepancy: {  normalized_discrepancy.item()}")


            # Also log discrepancy between projections
            writer.add_scalar("RestrictionMaps/Discrepancy", discrepancy.mean().item(), epoch)
            writer.add_scalar("RestrictionMaps/normalized_discrepancy", normalized_discrepancy.item(), epoch)
            writer.add_scalar("RestrictionMaps/squared_discrepancy", torch.sum(discrepancy**2).item(), epoch)

    return acc_1, test_loss_1/len(data_loader_test_1), acc_2, test_loss_2/len(data_loader_test_2)




def evaluate_models_testing(
        loss_1,
        loss_2,
        model_1, 
        model_2, 
        P12,
        P21,
        data_loader_test_1,
        data_loader_test_2,
        device
        ):
    
 

    model_1.eval()
    model_2.eval()

    correct_1, total_1, test_loss_1 = 0, 0, 0.0
    correct_2, total_2, test_loss_2= 0, 0, 0.0

    # EXTRACT DATA FROM MODELS 
    
    model_1_projection_list = []
    model_2_projection_list = []

    def hook_model1(module, input, output):
        # output here is fc1(x) (pre-ReLU), so apply ReLU manually
        flattened = output.view(output.size(0), -1)
        model_1_projection_list.append(flattened.detach().cpu())
            

    def hook_model2(module, input, output):
            # output here is fc1(x) (pre-ReLU), so apply ReLU manually
        flattened = output.view(output.size(0), -1)
        model_2_projection_list.append(flattened.detach().cpu())

    # Register the hook
    hook1 = model_1.conv_block3.register_forward_hook(hook_model1)
    hook2 = model_2.conv_block3.register_forward_hook(hook_model2)
    
    

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

        acc_2 = correct_2 / total_2
        logger.info(f"Model 2 - Loss: {test_loss_2/len(data_loader_test_2):.4f}, Acc: {acc_2:.4f}")


        # Remove hooks after evaluation
        hook1.remove()
        hook2.remove()

        # ---- Verify discrepancy between restriction maps ----
        # Concatenate all batches: [num_samples, fc1_dim]
        projection_all_model_1 = torch.cat(model_1_projection_list, dim=0).to(device) 
        projection_all_model_2 = torch.cat(model_2_projection_list, dim=0).to(device)  


    return acc_1, test_loss_1/len(data_loader_test_1), acc_2, test_loss_2/len(data_loader_test_2), projection_all_model_1, projection_all_model_2