import torch
import torch.nn as nn
from blocks import Encoder, SegmentationDecoder, Rec1, Rec2


class Net(nn.Module):
    def __init__(self, args, in_c=4, out_c=3):
        super(Net, self).__init__()
        
        self.encoder = Encoder(in_c)
        self.rec1 = Rec1(latent_dim=128)
        self.rec2 = Rec2(latent_dim=128)
        
        self.Decoder1 = SegmentationDecoder(512, out_c)   
        
        self.Decoder2 = SegmentationDecoder(512, out_c)     
        self.s_gap1 = nn.AdaptiveAvgPool2d(1)
        self.s_gap2 = nn.AdaptiveAvgPool2d(1)
        self.t_gap1 = nn.AdaptiveAvgPool2d(1)
        self.t_gap2 = nn.AdaptiveAvgPool2d(1)

    def forward(self, S=None, T=None, is_train=True):
        
        if is_train:
            s_f1, s_f2 = self.encoder(S)
            b_f1, c_f1, _, _ = s_f1.shape
            b_f2, c_f2, _, _ = s_f2.shape
        t_f1, t_f2 = self.encoder(T)
        
        if is_train:
            
            s_f1_gap = self.s_gap1(s_f1).view(b_f1, c_f1)
            s_f2_gap = self.s_gap2(s_f2).view(b_f2, c_f2)
            t_f1_gap = self.t_gap1(t_f1).view(b_f1, c_f1)
            t_f2_gap = self.t_gap2(t_f2).view(b_f2, c_f2)
            
            s_rec1, t_rec1 = self.rec1(s_f1, t_f1)
            s_rec2, t_rec2 = self.rec2(s_f2, t_f2)
            
            _, _, _, _, s_all = self.Decoder2(s_f2)
            d2_t_f1, d2_t_f1, d2_t_f1, d2_t_f1, t_all = self.Decoder2(t_f2)
            
            _, _, _, _, pred_s = self.Decoder1(s_f2)

        d1_t_f1, d1_t_f2, d1_t_f3, d1_t_f4, pred_t = self.Decoder1(t_f2)
        
        if is_train:
            return s_f1_gap, t_f1_gap, s_f2_gap, t_f2_gap, s_rec1, t_rec1, s_rec2, t_rec2, [d1_t_f1, d1_t_f2, d1_t_f3, d1_t_f4], [d2_t_f1, d2_t_f1, d2_t_f1, d2_t_f1], s_all, t_all, pred_s, pred_t
        else:
            return pred_t


if '__main__' == __name__:
    # P
    s = torch.randn((2, 4, 160, 160))
    t = torch.randn((2, 4, 160, 160))
    model = Net()
    a, b, c, d, e, f, g, h, i, j = model(s, t)
    print(a.shape, b.shape, c.shape, d.shape, e.shape, f.shape, g.shape, h.shape, i.shape, j.shape)
