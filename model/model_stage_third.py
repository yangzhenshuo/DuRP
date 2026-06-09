import torch.nn as nn
from model.model_stage_one import PRNet
from model.model_stage_second import IRNet


class MSPRNet(nn.Module):
    """
        Define the baseline network to predict K, P_D and theta_D
    """

    def __init__(self):
        super(MSPRNet, self).__init__()
        self.polModel = PRNet()
        self.IRModel = IRNet()
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
        K, m_A, m_D = self.polModel(I_alpha, m_I)
        A_inf, R_hat = self.IRModel(I_alpha, m_I, m_A, m_D)
        return m_A, m_D, A_inf, R_hat
        