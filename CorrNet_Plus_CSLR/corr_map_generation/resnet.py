import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import cv2
import numpy as np
import os
__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, clusters=1):
        super().__init__()
        #self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.clusters = clusters
        self.query = nn.Parameter(torch.rand(self.clusters, 1, embed_dim), requires_grad=True)

    def forward(self, x):
        N, C, T, H, W= x.shape
        x = x.flatten(start_dim=3).permute(3, 0, 2, 1).reshape(-1, N*T, C).contiguous()  # NCTHW -> (HW)(NT)C
        #x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)(NT)C
        #x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)(NT)C
        x, _ = F.multi_head_attention_forward(
            #query=x[:1], key=x, value=x,
            query=self.query.repeat(1,N*T,1), key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.view(self.clusters,N,T,C).contiguous().permute(1,3,2,0) #PNTC->NCTP

class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size=9, window_stride=1, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (window_size + (window_size-1) * (window_dilation-1) - 1) // 2
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1),
                                dilation=(self.window_dilation, 1),
                                stride=(self.window_stride, 1),
                                padding=(self.padding, 0))

    def forward(self, x):
        # Input shape: (N,C,T,H,W), out: (N,C,T,V*window_size)
        N, C, T, H, W = x.shape
        x = x.view(N, C, T, H*W)
        x = self.unfold(x)  #(N, C*Window_Size, T, H*W)
        # Permute extra channels from window size to the graph dimension; -1 for number of windows
        x = x.view(N, C, self.window_size, T, H, W).permute(0,1,3,2,4,5).reshape(N, C, T, self.window_size, H, W).contiguous()# NCTSHW
        return x

class Temporal_weighting(nn.Module):
    def __init__(self, input_size ):
        super().__init__()
        hidden_size = input_size//16
        self.conv_transform = nn.Conv1d(input_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.conv_back = nn.Conv1d(hidden_size, input_size, kernel_size=1, stride=1, padding=0)
        self.num = 3
        self.conv_enhance = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=int(i+1), groups=hidden_size, dilation=int(i+1)) for i in range(self.num)
        ])
        self.weights = nn.Parameter(torch.ones(self.num) / self.num, requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_transform(x.mean(-1).mean(-1))
        aggregated_out = 0
        for i in range(self.num):
            aggregated_out += self.conv_enhance[i](out) * self.weights[i]
        out = self.conv_back(aggregated_out)
        return x*(F.sigmoid(out.unsqueeze(-1).unsqueeze(-1))-0.5) * self.alpha

class Get_Correlation(nn.Module):
    def __init__(self, channels, neighbors=3, agg_mode='weighted'):
        super().__init__()
        reduction_channel = channels//16

        self.down_conv2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.neighbors = neighbors
        self.clusters = 1
        # used for learnable weighted-sum aggregation across neighbor positions
        self.weights2 = nn.Parameter(torch.ones(self.neighbors*2) / (self.neighbors*2), requires_grad=True)
        self.unfold = UnfoldTemporalWindows(2*self.neighbors+1)
        self.weights3 = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.weights4 = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.attpool = AttentionPool2d(spacial_dim=None, embed_dim=channels, num_heads=1, clusters=self.clusters)
        self.mlp = nn.Sequential(nn.Conv3d(channels, reduction_channel, kernel_size=1),
                                 nn.GELU(),
                                nn.Conv3d(reduction_channel, channels, kernel_size=1),)

        # For generating aggregated_x with multi-scale conv
        self.down_conv = nn.Conv3d(channels, reduction_channel, kernel_size=1, bias=False)
        self.spatial_aggregation1 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,1,1), groups=reduction_channel)
        self.spatial_aggregation2 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,2,2), dilation=(1,2,2), groups=reduction_channel)
        self.spatial_aggregation3 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,3,3), dilation=(1,3,3), groups=reduction_channel)
        self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.conv_back = nn.Conv3d(reduction_channel, channels, kernel_size=1, bias=False)

        # aggregation mode for multi-frame correlation maps
        # 'weighted' = learnable weighted sum across reference frames (Option B)
        # 'concat_conv' = concatenate affinities and apply a 1x1 conv (Option A)
        self.agg_mode = agg_mode
        if self.agg_mode == 'concat_conv':
            # conv across the neighbor dimension L (=2*neighbors)
            # use kernel_size equal to L to collapse the neighbor dimension
            self.agg_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2*self.neighbors, bias=False)

    def forward(self, x, return_affinity=False):
        N, C, T, H, W = x.shape

        # compute a validity mask that indicates which neighbor positions correspond to real frames
        # this lets us "use only available frames and normalize the aggregation weights" at boundaries
        ones = torch.ones((N, 1, T, H, W), dtype=x.dtype, device=x.device)
        mask_unfold = self.unfold(ones)  # N,1,T,window_size,H,W
        mask_concat = torch.concat([mask_unfold[:,:,:,:self.neighbors], mask_unfold[:,:,:,self.neighbors+1:]], 3)  # N,1,T,2*neighbors,H,W
        # sum over spatial dims to detect whether that neighbor position existed for each (N,T,neighbor)
        mask_sum = mask_concat.sum(dim=(4,5))  # N,1,T,2*neighbors
        valid_indicator = (mask_sum > 0).float()  # 1 if available, 0 if out-of-bounds

        x_mean = x.mean(3, keepdim=True).mean(4, keepdim=False)
        x_max = x.max(-1, keepdim=False)[0].max(-1, keepdim=True)[0]
        x_att = self.attpool(x) #NCTP
        x2 = self.down_conv2(x)
        upfold = self.unfold(x2)  # N,C,T,window_size,H,W
        # remove center frame from window to form reference set of length L=2*neighbors
        upfold = torch.concat([upfold[:,:,:,:self.neighbors], upfold[:,:,:,self.neighbors+1:]],3)  # N,C,T,L,H,W

        # apply per-neighbor learnable weight template but mask and renormalize to only available frames
        L = 2 * self.neighbors
        # expand the template weights to broadcast shape (N,1,T,L)
        template_w = self.weights2.view(1, 1, 1, L)
        valid = valid_indicator  # N,1,T,L
        # zero out template where invalid and renormalize across neighbors per (N,T)
        weighted = template_w * valid  # N,1,T,L
        denom = weighted.sum(dim=-1, keepdim=True)  # N,1,T,1
        denom = denom + 1e-6
        norm_w = weighted / denom  # N,1,T,L

        # flatten spatial dims for efficient einsum usage later: produce key of shape (N,C,T,L)
        upfold = upfold.view(N, C, T, L, H, W).contiguous()  # already in this shape but ensure
        # keep a spatial-flattened version for computing spatial affinities for visualization
        key_spatial = upfold.view(N, C, T, L * H * W)  # N,C,T,L*H*W
        # aggregate spatially (sum) so key reflects neighbor-frame level features (consistent with earlier behavior)
        key = upfold.sum(dim=(4,5))  # N,C,T,L

        x_mean = x_mean*self.weights4[0] + x_max*self.weights4[1] + x_att*self.weights4[2]

        # clustering / correlation computation
        # query: x_mean -> (N, C, T, P) (P==self.clusters)
        # key: key -> (N, C, T, L)  where L == 2*self.neighbors
        def clustering(query, key, key_spatial):
            # spatial affinities: (N, T, P, L*H*W)
            affinities_spatial = torch.einsum('bctp,bctl->btpl', query, key_spatial)
            # reshape to per-neighbor spatial maps: (N, T, P, L, H, W)
            affinities_spatial_reshaped = affinities_spatial.view(N, -1, self.clusters, L, H, W)
            # sum spatially to get scalar affinity per neighbor: (N, T, P, L)
            affinities_per_neighbor = affinities_spatial_reshaped.sum(dim=(4,5))

            aff = F.sigmoid(affinities_per_neighbor) - 0.5  # range [-0.5, 0.5]

            # apply validity mask to affinities so out-of-bounds positions do not contribute
            # valid_indicator: N,1,T,L -> reshape to N,T,1,L to match aff
            valid_perm = valid_indicator.squeeze(1).unsqueeze(2)  # N,T,1,L
            aff = aff * valid_perm  # zero-out invalid

            if self.agg_mode == 'weighted':
                # norm_w: N,1,T,L -> reshape to N,T,1,L for affinity weighting and N,1,T,L for key weighting
                w_aff = norm_w.squeeze(1).unsqueeze(2)  # N,T,1,L
                w_key = norm_w.unsqueeze(1)  # N,1,T,L
                # aggregate key across L with learned & normalized weights -> (N, C, T)
                key_sum = (key * w_key).sum(-1)
                # aggregate affinity across L into a scalar per (N,T,P)
                aff_sum = (aff * w_aff).sum(-1)
                # produce output features: (N, C, T, P)
                out = key_sum.unsqueeze(-1) * aff_sum.unsqueeze(1)
                return out, affinities_spatial_reshaped

            elif self.agg_mode == 'concat_conv':
                # aff shape -> (N, T, P, L)
                B, Tt, P, Ltmp = aff.shape
                # prepare affinity and mask tensors for Conv1d: (B*T*P, 1, L)
                aff_reshaped = aff.reshape(B*Tt*P, 1, Ltmp)
                mask_r = valid_indicator.squeeze(1)  # N,T,L
                mask_exp = mask_r.unsqueeze(2).repeat(1,1,P,1)  # N,T,P,L
                mask_reshaped = mask_exp.reshape(B*Tt*P, 1, Ltmp)
                # zero-out invalid positions (already zeroed in aff, but ensure mask used for normalization)
                aff_reshaped = aff_reshaped * mask_reshaped
                # apply 1D conv across neighbor dimension to aggregate -> (B*T*P,1,1)
                agg = self.agg_conv(aff_reshaped)  # (B*T*P,1,1)
                # normalize by number of available neighbors for that (N,T) to avoid bias from padding
                denom = mask_reshaped.sum(dim=2, keepdim=True)  # (B*T*P,1,1)
                denom = denom + 1e-6
                agg = agg / denom
                agg = agg.view(B, Tt, P, 1)  # (N, T, P, 1)
                # remove last dim and prepare for broadcasting: (N, T, P) -> (N, 1, T, P)
                agg_scalar = agg.squeeze(-1).contiguous()  # (N, T, P)
                agg_b = agg_scalar.unsqueeze(1)  # (N, 1, T, P)
                # simple key aggregation: weighted sum using normalized template weights -> (N, C, T)
                key_sum = (key * norm_w.unsqueeze(1)).sum(-1)  # (N, C, T)
                # multiply: (N, C, T, 1) * (N, 1, T, P) -> (N, C, T, P)
                out = key_sum.unsqueeze(-1) * agg_b
                return out, affinities_spatial_reshaped

            else:
                # fallback to original behaviour: use affinities to weight key directly
                out = torch.einsum('bctl,btpl->bctp', key, aff)
                return out, affinities_spatial_reshaped

        x_mean, affinities = clustering(x_mean, key, key_spatial)
        features = x_mean.view(N, C, T, self.clusters, 1)

        x_down = self.down_conv(x)
        aggregated_x = self.spatial_aggregation1(x_down)*self.weights[0] + self.spatial_aggregation2(x_down)*self.weights[1] \
                    + self.spatial_aggregation3(x_down)*self.weights[2]
        aggregated_x = self.conv_back(aggregated_x)
        
        features = features * (F.sigmoid(aggregated_x)-0.5)
        if not return_affinity:
            return features
        else:
            return features, affinities[0,:,0].view(-1, 2*self.neighbors, H, W)  #T(2*neighbors)HW 

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        stride=(1,stride,stride),
        padding=(0,1,1),
        bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, corr_neighbors=3, corr_agg_mode='concat_conv'):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # use corr_neighbors and corr_agg_mode from config/kwargs
        self.corr2 = Get_Correlation(self.inplanes, neighbors=corr_neighbors, agg_mode=corr_agg_mode)
        self.temporal_weight2 = Temporal_weighting(self.inplanes)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.corr3 = Get_Correlation(self.inplanes, neighbors=corr_neighbors, agg_mode=corr_agg_mode)
        self.temporal_weight3 = Temporal_weighting(self.inplanes)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.corr4 = Get_Correlation(self.inplanes, neighbors=corr_neighbors, agg_mode=corr_agg_mode)
        self.temporal_weight4 = Temporal_weighting(self.inplanes)
        self.alpha = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1,stride,stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, dataset):
        N, C, T, H, W = x.size()
        vid = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x) 
        x = x + self.corr2(x) * self.alpha[0]
        x = x + self.temporal_weight2(x)
        x = self.layer3(x)

        print(f'self.alpha: {self.alpha}')
        update_feature, affinities = self.corr3(x, return_affinity=True)  #bcthw, shw
        x = x + update_feature * self.alpha[1]
        show_corr_img(vid[0].permute(1,0,2,3), affinities, out_dir=f'./corr_map_layer3', clear_folder=True, dataset=dataset) #tchw, t(2*neighbors)hw

        x = x + self.temporal_weight3(x)
        x = self.layer4(x)
        x = x + self.corr4(x) * self.alpha[2]
        x = x + self.temporal_weight4(x)
        
        x = x.transpose(1,2).contiguous()
        x = x.view((-1,)+x.size()[2:]) #bt,c,h,w

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) #bt,c
        x = self.fc(x) #bt,c

        return x

def show_corr_img(img, affinities, out_dir='./corr_map', clear_folder=False, dataset='phoenix2014'):  # img: chw, feature_map: chw, grads: chw3
    affinities = affinities.cpu().data.numpy()
    if clear_folder:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            import shutil
            shutil.rmtree(out_dir)
            os.makedirs(out_dir)

    predefined_padding = 6 # Note that there are 6 paddings in advance in the left/right
    T, S, H, W = affinities.shape	
    neighbors = S//2	
    for t in range(predefined_padding, T-predefined_padding+1):
        current_dir = out_dir + '/' + f'timestep_{t-predefined_padding}'
        os.makedirs(current_dir)
        for i in range(S):
            if 'phoenix' in dataset:
                out_cam = affinities[t,i]  # only set as negative when alpha is positive for the layer
            else:
                out_cam = -affinities[t,i] 
            out_cam = out_cam - np.min(out_cam)
            out_cam = out_cam / (1e-7 + out_cam.max())
            out_cam = cv2.resize(out_cam, (img.shape[2], img.shape[3]))
            out_cam = (255 * out_cam).astype(np.uint8)
            heatmap = cv2.applyColorMap(out_cam, cv2.COLORMAP_JET)
            # img[neighbors] is the current image
            if i<neighbors:
                cam_img = np.float32(heatmap) / 255 + (img[t-(neighbors-i)]/2+0.5).permute(1,2,0).cpu().data.numpy()
            else:
                cam_img = np.float32(heatmap) / 255 + (img[t+(i-neighbors)+1]/2+0.5).permute(1,2,0).cpu().data.numpy()
            cam_img = cam_img/np.max(cam_img)
            cam_img = np.uint8(255 * cam_img)
            # img[neighbors] is the current image
            if i<neighbors:
                cv2.imwrite(f'{current_dir}/corr_map_{i}.jpg', cam_img)
            else:
                cv2.imwrite(f'{current_dir}/corr_map_{i+1}.jpg', cam_img)
        current_img = (img[t]/2+0.5).permute(1,2,0).cpu().data.numpy()
        current_img = current_img/np.max(current_img)
        current_img = np.uint8(255 * current_img)
        #interval = img.shape[2]//H
        #current_img[i*interval:(i+1)*interval, j*interval:(j+1)*interval,:] = np.array([0,0,255])  #red
        cv2.imwrite(f'{current_dir}/corr_map_{neighbors}_current.jpg', current_img)

def resnet18(**kwargs):
    """Constructs a ResNet-18 based model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'], map_location=torch.device('cpu'))
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)  
    model.load_state_dict(checkpoint, strict=False)
    del checkpoint
    import gc
    gc.collect()
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def test():
    net = resnet18()
    y = net(torch.randn(1,3,224,224))
    print(y.size())

#test()