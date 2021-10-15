import torch
import torch.nn as nn


def sparse_colmun(x):
    # L2,1
    x = x * x
    x = torch.sum(x, 0)
    x = torch.sqrt(x)
    return torch.sum(x)

def norm_inf(x):
    x = torch.abs(x)
    x = torch.max(x, 0)[0]
    #print(x.size())
    return torch.sum(x)

class ALDSloss(nn.Module):
    def __init__(self):
        super(ALDSloss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss()
        self.MSELoss = nn.MSELoss(reduction='sum')

    def forward(self, S, S_prime, Z, Z_prime, X, X_prime, Q, lamb):

        # The Reconstructed Loss
        loss_recS = self.MSELoss(S, S_prime)
        norm_q = norm_inf(Q)
        loss_S = loss_recS + lamb*norm_q

        loss_Z = self.MSELoss(Z, Z_prime)
        loss_r = self.MSELoss(X, X_prime)
        
        loss = {
            'loss_S': loss_S,
            'loss_recS': loss_recS,
            'norm_q': norm_q,
            'loss_Z': loss_Z,
            'loss_r': loss_r
        }

        return loss

"""
    Sort the samples
    
    net.eval()
    parameter = net.state_dict()
    C1 = parameter['coeff'].cpu().detach().numpy()
    C2 = parameter['coeff_cluster'].cpu().detach().numpy()

    C1 = C1 * C1
    C2 = C2 * C2
    C1 = (C1 - np.min(C1)) / (np.max(C1) - np.min(C1))
    C2 = (C2 - np.min(C2)) / (np.max(C2) - np.min(C2))

    s1 = np.sum(C1, 0)
    s2 = np.sum(C2, 0)
    s_ours = np.argsort(-(s1+s2))
"""