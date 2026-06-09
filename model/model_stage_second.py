import torch
import torch.nn as nn
from model.common import ALNetwork, SRRNetwork


class IRNet(nn.Module):
    """
        Define the baseline network to predict K, P_D and theta_D
    """

    def __init__(self, in_chans=3, embed_dim=32, n_downsampling=3, n_blocks=3):
        super(IRNet, self).__init__()
        # for airlight intensity
        self.Es_airintensity = ALNetwork(in_chans, embed_dim, n_downsampling, n_blocks)
        self.Re_scene = SRRNetwork(in_chans, embed_dim, n_downsampling, n_blocks)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1, 0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([p.numel() for p in model_parameters])
        return params
        
    def forward(self, I_alpha, m_I, m_A, m_D):
        # Preprocessing
        I0, I45, I90, I135 = torch.split(I_alpha, 3, dim=1)
        S0_I = (I0 + I45 + I90 + I135) / 2
        S1_I = I0 - I90
        S2_I = I45 - I135
        I_max = (S0_I + torch.sqrt(S1_I**2 + S2_I**2)) * 0.5
        I_min = (S0_I - torch.sqrt(S1_I**2 + S2_I**2)) * 0.5

        input_concat = torch.cat([I_alpha, I_max, I_min], dim=1)
        A_inf = self.Es_airintensity(input_concat)
        # multiply 2 because data range
        I_un = (I0 + I45 + I90 + I135) / 4
        R_hat = torch.abs((m_A-m_I)*I_un*A_inf/((m_A-m_D)*A_inf+(m_D-m_I)*I_un*2+1e-9))
        R_hat = torch.clamp(R_hat, 0, 1)
        # refine
        R = self.Re_scene(R_hat, I_alpha)
        return A_inf, R