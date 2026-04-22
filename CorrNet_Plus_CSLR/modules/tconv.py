import pdb
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F


class MS_TCN_Block(nn.Module):
    """Multi-Scale Temporal Convolution block inspired by USTM.

    Runs n_dilations+2 parallel 1D branches over the time dimension:
      - 2 branches are 1x1 convolutions for channel dimensionality reduction/mixing
      - n_dilations branches are dilated 3x1 convolutions (dilation=1,2,...,n_dilations)
        each with causal-symmetric padding so that T is preserved

    All branches are concatenated, then projected back to hidden_size via a 1x1 conv.
    T (sequence length) is never changed by this block — that is left to TLP as before.

    Args:
        input_size  (int): input channel dimension C_in (512 from ResNet)
        hidden_size (int): output channel dimension (1024, matching BiLSTM input)
        n_dilations (int): number of dilated branches (default 4, matching USTM)
    """
    def __init__(self, input_size: int, hidden_size: int, n_dilations: int = 4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_dilations = n_dilations

        # 2 dimensionality-reduction branches (1x1 conv, no temporal receptive field)
        self.reduce_branches = nn.ModuleList([
            nn.Conv1d(input_size, input_size, kernel_size=1)
            for _ in range(2)
        ])

        # n_dilations dilated branches, each kernel_size=3 with padding to keep T
        self.dilated_branches = nn.ModuleList([
            nn.Conv1d(
                input_size, hidden_size,
                kernel_size=3,
                dilation=d + 1,
                padding=d + 1,   # padding == dilation keeps T intact for k=3
            )
            for d in range(n_dilations)
        ])

        # total concatenated channels: input_size*2 + hidden_size*n_dilations
        merged_channels = input_size * 2 + hidden_size * n_dilations

        # project merged branches back to hidden_size
        self.merge = nn.Sequential(
            nn.Conv1d(merged_channels, hidden_size, kernel_size=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T]
        parts = [branch(x) for branch in self.reduce_branches]   # each [B, C_in, T]
        parts += [branch(x) for branch in self.dilated_branches]  # each [B, hidden, T]
        out = torch.cat(parts, dim=1)  # [B, merged_channels, T]
        return self.merge(out)         # [B, hidden_size, T]

class Temporal_LiftPool(nn.Module):
    def __init__(self, input_size, kernel_size=2):
        super(Temporal_LiftPool, self).__init__()
        self.kernel_size = kernel_size
        self.predictor = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, stride=1, padding=1, groups=input_size), 
            nn.ReLU(inplace=True),   
            nn.Conv1d(input_size, input_size, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),    
                                    )

        self.updater = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, stride=1, padding=1, groups=input_size),
            nn.ReLU(inplace=True),   
            nn.Conv1d(input_size, input_size, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),    
                                    )
        self.predictor[2].weight.data.fill_(0.0)
        self.updater[2].weight.data.fill_(0.0)
        self.weight1 = Local_Weighting(input_size)
        self.weight2 = Local_Weighting(input_size)

    def forward(self, x):
        B, C, T= x.size()
        Xe = x[:,:,:T:self.kernel_size]
        Xo = x[:,:,1:T:self.kernel_size]
        d = Xo - self.predictor(Xe)
        s = Xe + self.updater(d)
        loss_u = torch.norm(s-Xo, p=2)
        loss_p = torch.norm(d, p=2)
        s = torch.cat((x[:,:,:0:self.kernel_size], s, x[:,:,T::self.kernel_size]),2)
        return self.weight1(s)+self.weight2(d), loss_u, loss_p

class Local_Weighting(nn.Module):
    def __init__(self, input_size ):
        super(Local_Weighting, self).__init__()
        self.conv = nn.Conv1d(input_size, input_size, kernel_size=5, stride=1, padding=2)
        self.insnorm = nn.InstanceNorm1d(input_size, affine=True)
        self.conv.weight.data.fill_(0.0)

    def forward(self, x):
        out = self.conv(x)
        return x + x*(F.sigmoid(self.insnorm(out))-0.5)

class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1, mstcn_num_layers=4, mstcn_hidden_size=1024, mstcn_kernel_size=3):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        # MS-TCN configuration parameters
        self.mstcn_num_layers = mstcn_num_layers
        self.mstcn_hidden_size = mstcn_hidden_size
        self.mstcn_kernel_size = mstcn_kernel_size

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
            self.strides = [0]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
            self.strides = [4,0]
        elif self.conv_type == 3:
            # M: MS-TCN block preserving temporal length, followed by two pooling ops (÷4 total)
            self.kernel_size = ['M', "P2", "P2"]
            self.strides = [4,0]

        self.temporal_conv = nn.ModuleList([])
        #nums = 0
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                self.temporal_conv.append(Temporal_LiftPool(input_size=input_sz, kernel_size=int(ks[1])))
            elif ks[0] == 'K':
                self.temporal_conv.append(
                    nn.Sequential(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0),
                    nn.BatchNorm1d(self.hidden_size),
                    nn.ReLU(inplace=True),
                    )
                )
            elif ks[0] == 'M':
                # instantiate MS-TCN block
                # MultiScale_TCN expects input shape (T, B, C) and returns dict
                self.temporal_conv.append(MultiScale_TCN(in_channels=input_sz, num_layers=self.mstcn_num_layers, kernel_size=self.mstcn_kernel_size, hidden_size=self.mstcn_hidden_size, num_classes=self.num_classes))

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, feat_len):
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len //= int(ks[1])
            else:
                # 'K' reduces temporal length by kernel-1; 'M' preserves temporal length
                if ks[0] == 'K':
                    feat_len -= int(ks[1]) - 1
                elif ks[0] == 'M':
                    pass
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = frame_feat
        loss_LiftPool_u = 0
        loss_LiftPool_p = 0
        i = 0
        tcn_logits = None
        for tempconv in self.temporal_conv:
            if isinstance(tempconv, Temporal_LiftPool):
                visual_feat, loss_u, loss_d = tempconv(visual_feat) #self.strides[i])
                i +=1
                loss_LiftPool_u += loss_u
                loss_LiftPool_p += loss_d
            elif isinstance(tempconv, MultiScale_TCN):
                # MultiScale_TCN expects (T, B, C)
                tcn_out = tempconv(visual_feat.permute(2,0,1).contiguous())
                # tcn_feat: (T, B, C) -> convert back to (B, C, T)
                tcn_feat = tcn_out['tcn_feat'].permute(1,2,0).contiguous()
                visual_feat = tcn_feat
                if 'tcn_logits' in tcn_out:
                    # tcn_logits is (T, B, num_classes)
                    tcn_logits = tcn_out['tcn_logits']
            else:
                visual_feat = tempconv(visual_feat)
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 \
            else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        outputs = {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1) if logits is not None else None,
            "feat_len": lgt.cpu(),
            "loss_LiftPool_u": loss_LiftPool_u,
            "loss_LiftPool_p": loss_LiftPool_p,
        }
        if tcn_logits is not None:
            outputs['tcn_logits'] = tcn_logits
            outputs['tcn_feat'] = tcn_out['tcn_feat']
            # report conv-level (tcn) temporal length per batch so losses can align properly
            T_conv = int(tcn_out['tcn_feat'].size(0))
            B_conv = int(tcn_out['tcn_feat'].size(1))
            outputs['conv_feat_len'] = torch.tensor([T_conv] * B_conv, dtype=torch.int32)
        return outputs


# Minimal MultiScale_TCN implementation that composes MS_TCN_Block layers and outputs tcn_feat and optional logits
class MultiScale_TCN(nn.Module):
    def __init__(self, in_channels, num_layers=4, kernel_size=3, hidden_size=1024, num_classes=-1):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        layers = []
        # initial projection from input channels to hidden_size
        if in_channels != hidden_size:
            layers.append(nn.Conv1d(in_channels, hidden_size, kernel_size=1))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU(inplace=True))

        for i in range(num_layers):
            # MS_TCN_Block operates on (B, C, T)
            layers.append(MS_TCN_Block(hidden_size, hidden_size, n_dilations=4))

        self.tcn = nn.Sequential(*layers)
        if self.num_classes != -1:
            self.cls_head = nn.Conv1d(self.hidden_size, self.num_classes, kernel_size=1)
        else:
            self.cls_head = None

    def forward(self, x):
        # x: (T, B, C) -> convert to (B, C, T) for Conv1d
        t, b, c = x.shape
        out = x.permute(1,2,0).contiguous()
        out = self.tcn(out)
        # out: (B, hidden, T) -> convert back to (T, B, hidden)
        tcn_feat = out.permute(2,0,1).contiguous()
        out_dict = {'tcn_feat': tcn_feat}
        if self.cls_head is not None:
            logits = self.cls_head(out) # (B, num_classes, T)
            out_dict['tcn_logits'] = logits.permute(2,0,1).contiguous() # (T, B, num_classes)
        return out_dict