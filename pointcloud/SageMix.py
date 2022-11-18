import torch
from emd_ import emd_module

class SageMix:
    def __init__(self, args, num_class=40):
        self.num_class = num_class
        self.EMD = emd_module.emdModule()
        self.sigma = args.sigma
        self.beta = torch.distributions.beta.Beta(torch.tensor([args.theta]), torch.tensor([args.theta]))

    
    def mix(self, xyz, label, saliency=None):
        """
        Args:
            xyz (B,N,3)
            label (B)
            saliency (B,N): Defaults to None.
        """        
        B, N, _ = xyz.shape
        idxs = torch.randperm(B)

        
        #Optimal assignment in Eq.(3)
        perm = xyz[idxs]
        
        _, ass = self.EMD(xyz, perm, 0.005, 500) # mapping
        ass = ass.long()
        perm_new = torch.zeros_like(perm).cuda()
        perm_saliency = torch.zeros_like(saliency).cuda()
        
        for i in range(B):
            perm_new[i] = perm[i][ass[i]]
            perm_saliency[i] = saliency[idxs][i][ass[i]]
        
        #####
        # Saliency-guided sequential sampling
        #####
        #Eq.(4) in the main paper
        saliency = saliency/saliency.sum(-1, keepdim=True)
        anc_idx = torch.multinomial(saliency, 1, replacement=True)
        anchor_ori = xyz[torch.arange(B), anc_idx[:,0]]
        
        #cal distance and reweighting saliency map for Eq.(5) in the main paper
        sub = perm_new - anchor_ori[:,None,:]
        dist = ((sub) ** 2).sum(2).sqrt()
        perm_saliency = perm_saliency * dist
        perm_saliency = perm_saliency/perm_saliency.sum(-1, keepdim=True)
        
        #Eq.(5) in the main paper
        anc_idx2 = torch.multinomial(perm_saliency, 1, replacement=True)
        anchor_perm = perm_new[torch.arange(B),anc_idx2[:,0]]
                
                
        #####
        # Shape-preserving continuous Mixup
        #####
        alpha = self.beta.sample((B,)).cuda()
        sub_ori = xyz - anchor_ori[:,None,:]
        sub_ori = ((sub_ori) ** 2).sum(2).sqrt()
        #Eq.(6) for first sample
        ker_weight_ori = torch.exp(-0.5 * (sub_ori ** 2) / (self.sigma ** 2))  #(M,N)
        
        sub_perm = perm_new - anchor_perm[:,None,:]
        sub_perm = ((sub_perm) ** 2).sum(2).sqrt()
        #Eq.(6) for second sample
        ker_weight_perm = torch.exp(-0.5 * (sub_perm ** 2) / (self.sigma ** 2))  #(M,N)
        
        #Eq.(9)
        weight_ori = ker_weight_ori * alpha 
        weight_perm = ker_weight_perm * (1-alpha)
        weight = (torch.cat([weight_ori[...,None],weight_perm[...,None]],-1)) + 1e-16
        weight = weight/weight.sum(-1)[...,None]

        #Eq.(8) for new sample
        x = weight[:,:,0:1] * xyz + weight[:,:,1:] * perm_new
        
        #Eq.(8) for new label
        target = weight.sum(1)
        target = target / target.sum(-1, keepdim=True)
        label_onehot = torch.zeros(B, self.num_class).cuda().scatter(1, label.view(-1, 1), 1)
        label_perm_onehot = label_onehot[idxs]
        label = target[:, 0, None] * label_onehot + target[:, 1, None] * label_perm_onehot 
        
        return x, label
    