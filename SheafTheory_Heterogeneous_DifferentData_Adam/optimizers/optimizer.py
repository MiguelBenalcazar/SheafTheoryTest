import torch
from torch.optim import Optimizer
from collections.abc import Iterable


# class model_optimizer(Optimizer):
#     def __init__(self, params, lr=0.01):
#         # params = model.parameters()
#         defaults = dict(lr=lr)
#         super(model_optimizer, self).__init__(params, defaults)

#     @torch.no_grad()  # disable gradient tracking during update
#     def step(self, closure=None):
#         """
#         Performs a single optimization step.
#         closure: optional, re-evaluates the model and returns the loss
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()

#         for group in self.param_groups:  # iterate over parameter groups
#             lr = group['lr']
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 # update rule: θ = θ - lr * ∇θ
#                 p.data = p.data - lr * p.grad.data
#         return loss


# class model_optimizer(Optimizer):
#     def __init__(self, 
#                  model_1:torch.nn.Module, 
#                  model_2:torch.nn.Module, 
#                  P12:torch.nn.Module, 
#                  P21:torch.nn.Module, 
#                  lr:int=0.01, 
#                  lambda_reg:int=0.01
#                  ):
#         # params = model.parameters()
#         defaults = dict(lr=lr)
#         super(model_optimizer, self).__init__(model_1.parameters(), defaults)

#         self.model_1 = model_1
#         self.model_2 = model_2
#         self.P12 = P12
#         self.P21 = P21                                                                                      
#         self.lambda_reg = lambda_reg

#     def flatten_parameters(self, model):
#         return torch.cat([p.view(-1) for p in model.parameters()])

#     @torch.no_grad()  # disable gradient tracking during update
#     def step(self, closure=None):
#         """
#         Performs a single optimization step.
#         closure: optional, re-evaluates the model and returns the loss
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()

#         # Flatten parameters
#         theta1_vec = self.flatten_parameters(self.model_1)
#         theta2_vec = self.flatten_parameters(self.model_2)
#         device = theta1_vec.device

#         # Compute regularization term
#         proj1 = self.P12(theta1_vec)
#         proj2 = self.P21(theta2_vec)
#         discrepancy = proj1 - proj2
#         reg_term = self.P12.weight.t() @ discrepancy  # same shape as theta1_vec


#         # Now update each parameter in-place
#         pointer = 0
#         for group in self.param_groups:  # iterate over parameter groups
#             lr = group['lr']
#             for p in group['params']:
#                 if p.grad is None:
#                     continue

#                 numel = p.numel()

#                 # Extract corresponding slice of reg_term
#                 reg_slice = reg_term[pointer:pointer+numel].view_as(p)
#                 pointer += numel

#                 # Update: θ = θ - lr * (∇θ + λ * reg_term)
#                 p.data = p.data - lr * (p.grad.data + self.lambda_reg * reg_slice)
#         return loss
    



class model_optimizer(Optimizer):
    def __init__(self, 
                 params,
                 lr:int=0.01, 
                 lambda_reg:int=0.01
                 ):
        # params = model.parameters()
        defaults = dict(lr=lr, lambda_reg=lambda_reg)
        super(model_optimizer, self).__init__(params, defaults)


    @torch.no_grad()  # disable gradient tracking during update
    def step(self, 
             model1_vec, 
             model2_vec, 
             P12, 
             P21, 
             closure=None):
        """
        Performs a single optimization step.
        closure: optional, re-evaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            loss = closure()



        # Compute regularization term
        discrepancy = P12(model1_vec ) - P21(model2_vec)
        reg_term = P12.weight.t() @ discrepancy  # same shape as theta1_vec


        # Now update each parameter in-place
        pointer = 0
        for group in self.param_groups:  # iterate over parameter groups
            lr = group['lr']
            lambda_reg = group['lambda_reg']
            for p in group['params']:
                if p.grad is None:
                    continue

                numel = p.numel()

                # Extract corresponding slice of reg_term
                reg_slice = reg_term[pointer:pointer+numel].view_as(p)
                pointer += numel

                # Update: θ = θ - lr * (∇θ + λ * reg_term)
                p.data = p.data - lr * (p.grad.data + lambda_reg * reg_slice)
        return loss
    

class restriction_maps_optimizer:
    def __init__(self, 
                 P12:torch.nn.Module, 
                 P21:torch.nn.Module, 
                 lr:int=0.01, 
                 lambda_reg:int=0.01
                 ):
       

        self.P12 = P12
        self.P21 = P21   
        self.lr = lr                                                                                   
        self.lambda_reg = lambda_reg


    @torch.no_grad()  # disable gradient tracking during update
    def step(self, theta1_vec, theta2_vec):
        """
        Manual update of P12 and P21 (no autograd).
        theta1_vec, theta2_vec: flattened parameter vectors of models
        """
        # Compute discrepancy
        proj1 = self.P12(theta1_vec)   # P12 θ1
        proj2 = self.P21(theta2_vec)   # P21 θ2

        discrepancy12 = proj1 - proj2
        discrepancy21 = proj2 - proj1


        # print(discrepancy12.shape)


        # theta1_vec = theta1_vec.view(-1,1)  # make column vector
        # theta2_vec = theta2_vec.view(-1,1)

        # # Apply updates
        # self.P12.weight -= self.lr * self.lambda_reg  * discrepancy12.view(-1,1) @  theta1_vec.T
        # self.P21.weight -= self.lr * self.lambda_reg  * discrepancy21.view(-1,1) @  theta2_vec.T

         # Apply updates
        self.P12.weight -= self.lr * self.lambda_reg  * discrepancy12.T @ theta1_vec
        self.P21.weight -= self.lr * self.lambda_reg  * discrepancy21.T @ theta2_vec


class restriction_maps_optimizer_P12:
    def __init__(self, 
                 P12:torch.nn.Module, 
                 P21:torch.nn.Module, 
                 lr:int=0.01, 
                 lambda_reg:int=0.01
                 ):
       

        self.P12 = P12
        self.P21 = P21   
        self.lr = lr                                                                                   
        self.lambda_reg = lambda_reg


    @torch.no_grad()  # disable gradient tracking during update
    def step(self, theta1_vec, theta2_vec):
        """
        Manual update of P12 and P21 (no autograd).
        theta1_vec, theta2_vec: flattened parameter vectors of models
        """
        # Compute discrepancy
        proj1 = self.P12(theta1_vec)   # P12 θ1
        proj2 = self.P21(theta2_vec)   # P21 θ2

        discrepancy12 = proj1 - proj2
        self.P12.weight -= self.lr * self.lambda_reg  * discrepancy12.T @ theta1_vec

class restriction_maps_optimizer_P21:
    def __init__(self, 
                 P12:torch.nn.Module, 
                 P21:torch.nn.Module, 
                 lr:int=0.01, 
                 lambda_reg:int=0.01
                 ):
       

        self.P12 = P12
        self.P21 = P21   
        self.lr = lr                                                                                   
        self.lambda_reg = lambda_reg


    @torch.no_grad()  # disable gradient tracking during update
    def step(self, theta1_vec, theta2_vec):
        """
        Manual update of P12 and P21 (no autograd).
        theta1_vec, theta2_vec: flattened parameter vectors of models
        """
        # Compute discrepancy
        proj1 = self.P12(theta1_vec)   # P12 θ1
        proj2 = self.P21(theta2_vec)   # P21 θ2

        discrepancy21 = proj2 - proj1
        self.P21.weight -= self.lr * self.lambda_reg  * discrepancy21.T @ theta2_vec
    