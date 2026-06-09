import torch
import torch.nn as nn
from model.common import PTNetwork, APNetwork, SPRNetwork

class PRNet(nn.Module):
    """
        Define the baseline network to predict K, P_D and theta_D
    """

    def __init__(self, in_chans=3, embed_dim=32, n_downsampling=3, n_blocks=3):
        super(PRNet, self).__init__()
        # for pol transimission
        self.Es_poltransmission = PTNetwork(in_chans, embed_dim, n_downsampling, n_blocks)
        # for airlight pol params
        self.Es_airpolarization = APNetwork(in_chans, embed_dim, n_downsampling, n_blocks)
        # refine scene pol params
        self.Re_scepolarization = SPRNetwork(in_chans, embed_dim, n_downsampling, n_blocks)
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
        
    def forward(self, I_alpha, m_I):
        # Preprocessing
        I0, I45, I90, I135 = torch.split(I_alpha, 3, dim=1)
        S0_I = (I0 + I45 + I90 + I135) / 2
        S1_I = I0 - I90
        S2_I = I45 - I135
        I_max = (S0_I + torch.sqrt(S1_I**2 + S2_I**2)) * 0.5
        I_min = (S0_I - torch.sqrt(S1_I**2 + S2_I**2)) * 0.5
        
        s1_Ip = (m_I.real + 1) / 2
        s2_Ip = (m_I.imag + 1) / 2
        # for polarized transmission map
        input_concat = torch.cat([I_alpha, I_max, I_min], dim=1)
        K = self.Es_poltransmission(input_concat)
        # for airlight S1 and S2
        m_A = self.Es_airpolarization(input_concat)
        # calculate the polarization of the transmission light by PRM
        m_D_hat = m_A + (m_I - m_A) * K
        # refine the scene polarization
        s1_Dp_hat = (m_D_hat.real + 1) / 2
        s2_Dp_hat = (m_D_hat.imag + 1) / 2
        input_Dp = torch.cat([s1_Dp_hat, s2_Dp_hat], dim=1)
        input_Ip = torch.cat([s1_Ip, s2_Ip], dim=1)
        m_D = self.Re_scepolarization(input_Dp, input_Ip)
        return K, m_A, m_D