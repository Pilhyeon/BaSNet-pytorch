import torch
import torch.nn as nn

class Filter_Module(nn.Module):
    def __init__(self, len_feature):
        super(Filter_Module, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=1,
                    stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1,
                    stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, F)        
        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        out = self.conv_1(out)
        out = self.conv_2(out)
        out = out.permute(0, 2, 1)
        # out: (B, T, 1)
        return out
        

class CAS_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(CAS_Module, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU()
        )
                
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes + 1, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )
        self.drop_out = nn.Dropout(p=0.7)

    def forward(self, x):
        # x: (B, T, F)
        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        out = self.conv_1(out)
        out = self.conv_2(out)
        out = self.drop_out(out)
        out = self.conv_3(out)
        out = out.permute(0, 2, 1)
        # out: (B, T, C + 1)
        return out

class BaS_Net(nn.Module):
    def __init__(self, len_feature, num_classes, num_segments):
        super(BaS_Net, self).__init__()
        self.filter_module = Filter_Module(len_feature)
        self.len_feature = len_feature
        self.num_classes = num_classes

        self.cas_module = CAS_Module(len_feature, num_classes)

        self.softmax = nn.Softmax(dim=1)

        self.num_segments = num_segments
        self.k = num_segments // 8
    

    def forward(self, x):
        fore_weights = self.filter_module(x)

        x_supp = fore_weights * x

        cas_base = self.cas_module(x)
        cas_supp = self.cas_module(x_supp)

        score_base = torch.mean(torch.topk(cas_base, self.k, dim=1)[0], dim=1)
        score_supp = torch.mean(torch.topk(cas_supp, self.k, dim=1)[0], dim=1)

        score_base = self.softmax(score_base)
        score_supp = self.softmax(score_supp)

        return score_base, cas_base, score_supp, cas_supp, fore_weights