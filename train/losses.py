import torch
import torch.nn as nn
import torch.nn.functional as F
      
class pol_loss(nn.Module):
    def __init__(self):
        super(pol_loss, self).__init__()
        # Define weights for L1 and L2 losses
        self.l1_weight = 1
        self.l2_weight = 2

    def forward(self, x, x_hat, y, y_hat, z, z_hat):
        # Compute K loss
        K = x
        K_hat = torch.clamp(1 / x_hat, max=1, min=0)  # Clamp values between 0 and 1
        k_l1_loss = F.l1_loss(K, K_hat) * self.l1_weight
        k_l2_loss = F.mse_loss(K, K_hat) * self.l2_weight
        k_loss = k_l1_loss + k_l2_loss

        # Compute mA loss
        S1_A = (y.real + 1) / 2  # Normalize real part of y
        S2_A = (y.imag + 1) / 2  # Normalize imaginary part of y
        S1_A_hat = (y_hat.real + 1) / 2  # Normalize real part of y_hat
        S2_A_hat = (y_hat.imag + 1) / 2  # Normalize imaginary part of y_hat
        A_l1_loss = (F.l1_loss(S1_A, S1_A_hat) + F.l1_loss(S2_A, S2_A_hat)) * self.l1_weight
        A_l2_loss = (F.mse_loss(S1_A, S1_A_hat) + F.mse_loss(S2_A, S2_A_hat)) * self.l2_weight
        A_loss = A_l1_loss + A_l2_loss

        # Compute mD loss
        S1_D = (z.real + 1) / 2  # Normalize real part of z
        S2_D = (z.imag + 1) / 2  # Normalize imaginary part of z
        S1_D_hat = (z_hat.real + 1) / 2  # Normalize real part of z_hat
        S2_D_hat = (z_hat.imag + 1) / 2  # Normalize imaginary part of z_hat
        D_l1_loss = (F.l1_loss(S1_D, S1_D_hat) + F.l1_loss(S2_D, S2_D_hat)) * self.l1_weight
        D_l2_loss = (F.mse_loss(S1_D, S1_D_hat) + F.mse_loss(S2_D, S2_D_hat)) * self.l2_weight
        D_loss = D_l1_loss + D_l2_loss

        # Compute total loss
        loss = D_loss * 5 + A_loss * 1 + k_loss * 1

        return loss
   
class img_loss(nn.Module):
    def __init__(self):
        super(img_loss, self).__init__()
        
        self.l1_weight = 1
        self.l2_weight = 2
        
        
    def forward(self, x, x_hat, y, y_hat):
        
        #### A_inf ####
        AInf_l1_loss = F.l1_loss(x, x_hat) * self.l1_weight
        AInf_l2_loss = F.mse_loss(x, x_hat) * self.l2_weight
        AInf_loss = AInf_l1_loss + AInf_l2_loss
        # ##### R ####
        R_l1_loss = F.l1_loss(y, y_hat) * self.l1_weight
        R_l2_loss = F.mse_loss(y, y_hat) * self.l2_weight
        R_loss = R_l1_loss + R_l2_loss
        #### total loss ####
        loss = AInf_loss * 1 + R_loss * 2
        return loss
    
class full_loss(nn.Module):
    def __init__(self):
        super(full_loss, self).__init__()
        
        self.l1_weight = 1
        self.l2_weight = 2
    def forward(self, x, x_hat, y, y_hat, z, z_hat, w, w_hat):
        #### m_D ####
        S1_D = (y.real+1)/2
        S2_D = (y.imag+1)/2
        S1_D_hat = (y_hat.real+1)/2
        S2_D_hat = (y_hat.imag+1)/2
        D_l1_loss = (F.l1_loss(S1_D, S1_D_hat) + F.l1_loss(S2_D, S2_D_hat)) * self.l1_weight
        D_l2_loss = (F.mse_loss(S1_D, S1_D_hat) + F.mse_loss(S2_D, S2_D_hat)) * self.l2_weight
        D_loss = D_l1_loss + D_l2_loss
        #### R ####
        R_l1_loss = F.l1_loss(w, w_hat) * self.l1_weight
        R_l2_loss = F.mse_loss(w, w_hat) * self.l2_weight
        R_loss = R_l1_loss + R_l2_loss
        #### total loss ####
        loss = D_loss *1 + R_loss * 1.2
        return loss  
    
          
def get_loss(opt):
    opt_loss = opt.loss.lower()
    if opt_loss == 'pol_loss':
        criterion = pol_loss()
    elif opt_loss == 'img_loss':
        criterion = img_loss()
    elif opt_loss == 'full_loss':
        criterion = full_loss()
    else:
        raise NotImplementedError('Loss [{:s}] is not found'.format(opt.loss))
    return criterion
