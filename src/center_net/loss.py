from torch import nn
import torch
from torch.nn import functional as F


def _tranpose_and_gather_feature(feature, ind):
    feature = feature.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] => [B, H, W, C]
    feature = feature.view(feature.size(0), -1, feature.size(3))  # [B, H, W, C] => [B, H x W, C]
    ind = ind[:, :, None].expand(ind.shape[0], ind.shape[1], feature.shape[-1])  # [B, num_obj] => [B, num_obj, C]
    feature = feature.gather(1, ind)  # [B, H x W, C] => [B, num_obj, C]
    return feature

class CenterNetLoss(nn.Module):
    def __init__(self):
        """This is the loss used to train the CenterNet model.
        L = Ldet + Loff + Lpull + Lpush
        """
        
        super(CenterNetLoss, self).__init__()

    def focal_loss(self, y_hat, y): 
        """This loss is used to train the CenterNet model.
        It is also known as the Ldet or neg_loss.
        """
        
        pos_inds = y == 1
        neg_inds = y < 1
        
        neg_weights = torch.pow(1 - y[neg_inds], 4)
        
        loss = 0
        for pred in y_hat:
            pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
            pos_pred = pred[pos_inds]
            neg_pred = pred[neg_inds]

            pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
            neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

            num_pos = pos_inds.float().sum()
            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()

            if pos_pred.nelement() == 0:
                loss = loss - neg_loss
            else:
                loss = loss - (pos_loss + neg_loss) / num_pos
        
        return loss / len(y_hat)
        
    def l1_loss(self, regs, gt_regs, mask):
        """This loss is used to train the CenterNet model.
        It is also known as the Loff or reg_loss.
        """
        num = mask.float().sum() + 1e-4
        mask = mask[:, :, None].expand_as(gt_regs)  # [B, num_obj, 2]
        loss = sum([F.smooth_l1_loss(r[mask], gt_regs[mask], reduction='sum') / num for r in regs])

        return loss / len(regs)
        
    def ae_loss(self, embd0s, embd1s, mask):
        """This loss is used to train the CenterNet model.
        It is also known as the Lpull/Lpush or ae_loss.
        """
        num = mask.sum(dim=1, keepdim=True).float()  # [B, 1]

        pull, push = 0, 0
        for embd0, embd1 in zip(embd0s, embd1s):
            embd0 = embd0.squeeze()  # [B, num_obj]
            embd1 = embd1.squeeze()  # [B, num_obj]

            embd_mean = (embd0 + embd1) / 2

            embd0 = torch.pow(embd0 - embd_mean, 2) / (num + 1e-4)
            embd0 = embd0[mask].sum()
            embd1 = torch.pow(embd1 - embd_mean, 2) / (num + 1e-4)
            embd1 = embd1[mask].sum()
            pull += embd0 + embd1

            push_mask = (mask[:, None, :] + mask[:, :, None]) == 2  # [B, num_obj, num_obj]
            dist = F.relu(1 - (embd_mean[:, None, :] - embd_mean[:, :, None]).abs(), inplace=True)
            dist = dist - 1 / (num[:, :, None] + 1e-4)  # substract diagonal elements
            dist = dist / ((num - 1) * num + 1e-4)[:, :, None]  # total num element is n*n-n
            push += dist[push_mask].sum()

        return pull / len(embd0s), push / len(embd0s)
    
    def forward(self, outputs, batch):
        hmap_tl, hmap_br, hmap_ct, embd_tl, embd_br, regs_tl, regs_br, regs_ct = zip(*outputs)
        
        embd_tl = [_tranpose_and_gather_feature(e, batch['inds_tl']) for e in embd_tl]
        embd_br = [_tranpose_and_gather_feature(e, batch['inds_br']) for e in embd_br]
        regs_tl = [_tranpose_and_gather_feature(r, batch['inds_tl']) for r in regs_tl]
        regs_br = [_tranpose_and_gather_feature(r, batch['inds_br']) for r in regs_br]
        regs_ct = [_tranpose_and_gather_feature(r, batch['inds_ct']) for r in regs_ct]
        
        focal_loss = self.focal_loss(hmap_tl, batch['hmap_tl']) + \
                     self.focal_loss(hmap_br, batch['hmap_br']) + \
                     self.focal_loss(hmap_ct, batch['hmap_ct'])

        reg_loss = self.l1_loss(regs_tl, batch['regs_tl'], batch['ind_masks']) + \
                   self.l1_loss(regs_br, batch['regs_br'], batch['ind_masks']) + \
                   self.l1_loss(regs_ct, batch['regs_ct'], batch['ind_masks'])
                   
        pull_loss, push_loss = self.ae_loss(embd_tl, embd_br, batch['ind_masks'])
        
        return focal_loss + 0.1 * pull_loss + 0.1 * push_loss + reg_loss
        
        