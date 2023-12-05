# Based on Pointnet++
# FPS/SES/RS + Query Ball + LSTM/Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn

class QuantizationLayer(nn.Module):
    def __init__(self,dim=(128,128)):
        nn.Module.__init__(self)
        self.dim = dim

    def forward(self, xy):
        # input: xy-[B,N,2]
        # return: xy-[B,N,2],vox-[B,H,W]
        device = xy.device
        B,N,C = xy.shape
        W,H = self.dim                                     # [W,H]
        # 将xy中的坐标转换为非负整数；利用x,y,b,p等构建索引
        xy = (xy * (min(self.dim)-1)).int()                    # [B,N,2]
        x = xy[:,:,0].reshape(-1,)                         # [B*N]
        y = xy[:,:,1].reshape(-1,)                         # [B*N]
        b = torch.arange(B*N, dtype=torch.long).to(device) # [B*N]
        for bi in range(B):
            bi_idx = bi * torch.ones(N, dtype=torch.int32)
            b[bi*N:bi*N+N] = bi_idx

        # 设置计数的vox
        num_voxels = int(H*W*B)
        vox = xy[0,0,:].new_full([num_voxels,], fill_value=0).to(device)  # [B*H*W]

        # b,y,x 在vox 中的归宿 ; idx从右往上左就是 [B,H,W]
        idx = (x + W * y + W * H * b).long()
        values = torch.ones(B*N,dtype=torch.int32).to(device)

        vox.put_(idx, values, accumulate=True)
        vox = vox.view(-1, H, W)                                    # [B,H,W]

        return xy,vox



class Surface_Event_Sample(nn.Module):
    def __init__(self,dim=(128,128)):  # dim:W,H
        nn.Module.__init__(self)
        self.dim = dim
        self.quantization_layer = QuantizationLayer(dim=self.dim)

    def sample_batch(self,nevent,xy,vox,batch_sample_idx):
        B,_,_ = xy.shape

        for bi in range(B):
            bi_xy = xy[bi,:,:]          # [N,C]
            bi_vox = vox[bi,:,:]        # [H,W]
            bi_sample_idx = self.sample_single(nevent,bi_xy,bi_vox)  # [nevent,]
            batch_sample_idx[bi,:] = bi_sample_idx            # [B,nevent]

        return batch_sample_idx


    def sample_single(self,nevent,bi_xy,bi_vox):
        device = bi_xy.device
        bi_x = bi_xy[:, 0].long()
        bi_y = bi_xy[:, 1].long()
        # 注意：torch.sum(bi_vox) = N ; 而 torch.sum(bi_vox[bi_x,bi_y]) >> N
        # 独一无二（xy对应一个events）的events在序列中的索引
        idx_global_1 = torch.where(bi_vox[bi_y, bi_x] == 1)[0]
        # 冗余的（xy对应多个events）的events在序列中的索引
        idx_global_2 = torch.where(bi_vox[bi_y, bi_x] > 1)[0]
        #  在冗余的events中去重，并返回独一无二的xy的索引
        encode_xy = bi_x[idx_global_2] + bi_y[idx_global_2] * 10000
        _, local_idx = np.unique(encode_xy.cpu().numpy(), return_index=True)  # local_idx: 指向索引idx_global_2的索引
        local_idx = torch.from_numpy(local_idx).to(device)
        idx_global_3 = idx_global_2[local_idx]
        idx_unique = torch.cat([idx_global_1, idx_global_3], dim=0)

        idx_out = idx_unique[torch.randint(0, len(idx_unique), (nevent,), dtype=torch.long).to(device)]

        return idx_out


    def forward(self, xy, nevent):
        # xy -> vox: [B,N,C] -> [B,H,W]
        # 初始化
        B,N,C = xy.shape
        device = xy.device
        batch_sample_idx = torch.zeros([B, nevent], dtype=torch.long).to(device)  # 索引：[B,nevent]
        # 计算计数vox
        xy,vox = self.quantization_layer.forward(xy)      # vox:[B,H,W], xy: int32,[0,H/W-1]
        # 采样
        last_sample_idx = self.sample_batch(nevent,xy,vox,batch_sample_idx)       # vox:[B,nevent]
        # 还原xy
        xy = torch.true_divide(xy,min(self.dim))
        return xy,last_sample_idx


surface_event_sample = Surface_Event_Sample(dim=(256,256)) # W-H,X-Y


#   计算2个点集的欧式距离
def square_distance(src, dst):
    """
    输入: src_code: 原始点集, [B, N, 2]
         dst: 目标点集, [B, M, 2]
    输出: dist: 每个点的 square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


#   根据采样索引从原始events中取出采样events
def index_events(events,idx):
    """
    输入: events: input events data, [B, N, 2]
         idx: sample index data, [B,nevent,nsample]--指向索引
    输出: new_events:, indexed events data, [B,nevent,nsample,2]--指向坐标
    注：对于batch中的每个点集，FPS或Research Circle获得的索引不一样
    """
    device = events.device
    B = events.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)     # [1]*2=[1,1];view_shape=[B,1,1]
    repeat_shape = list(idx.shape)                   # [B,nevent,nsample]
    repeat_shape[0] = 1                              # repeat_shape=[1,nevent,nsample]
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) # batch_indices=[B,nevent,nsample]
    new_events = events[batch_indices, idx, :]       # [B,nevent,nsample,2]
    return new_events


#   FPS采样算法
def farthest_point_sample(xy,nevent):
    """
    输入: xy: AER events, [B, N, 2]
         nevent: 采样event数
    输出: centroids: 采样到的AER events index, [B,nevent]，这些点是key points,它们作为各自局部区域的centroid
    """
    device = xy.device
    B, N, C = xy.shape      # [32,8000,2]
    centroids = torch.zeros([B,nevent], dtype=torch.long).to(device)    # 索引：[32,3000]
    distance = torch.ones(B, N).to(device) * 1e10                       # 距离：[32,8000]，每个点距离其他所有点的总距离
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # batch中第1个点的索引（在整个点集中的索引）随机设置,[32,]
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(nevent):
        centroids[:, i] = farthest                                     # 第1个centroid的索引（在整个点集中的索引）随机设置,之后是通过计算获得的
        centroid = xy[batch_indices, farthest, :].view(B, 1, 2)        # 每个AER events的centriod坐标,[32,1,2]
        dist = torch.sum((xy - centroid) ** 2, -1)                     # AER events中每个点与centriod距离的总和，[32,8000]
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]                          # [32,]
    return centroids


#   Research Circle近邻圆形搜索算法
def research_circle(radius, nsample, xy, new_xy):
    """
    输入: radius: 局部邻域圆半径
         nsample: 在局部邻域内的最大取样点数
         xy: 所有原始AER events, [B, N, 2]
         new_xy: 查询点集, [B, S, 2]
    输出: group_idx: 每个局部区域内取样的点索引, [B, S, nsample]
    """
    device = xy.device
    B, N, C = xy.shape       # N表示原始点数
    _, S, _ = new_xy.shape   # S表示采样得到的Key Points数
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])    # [B, S, N]
    sqrdists = square_distance(new_xy, xy)                                                      # 计算查询点集和原始点集的中各2点之间的距离,[B, S, N]
    group_idx[sqrdists > radius ** 2] = N                                                       # 不在邻域圆内的点的索引，对应的值设置为N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]                                       # 截取前nsample个索引
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class Normalize(nn.Module):
    def __init__(self,channel,normalize="center"):
        super(Normalize, self).__init__()
        self.normalize = normalize
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1,1,1,channel]))

    def forward(self,new_xy,new_events,grouped_xy,grouped_events):
        B,_,_ = new_xy.shape
        grouped_events = torch.cat([grouped_events, grouped_xy],dim=-1)      # [B,S,K,2+D]
        if self.normalize =="center":
            mean = torch.mean(grouped_events, dim=2, keepdim=True)           # [B,S,1,2+D]
        if self.normalize =="anchor":
            mean = torch.cat([new_events,new_xy],dim=-1)
            mean = mean.unsqueeze(dim=-2)                                    # [B,S,1,2+D]
        std = torch.std((grouped_events-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
        grouped_events = (grouped_events-mean)/(std + 1e-5)
        new_events = self.affine_alpha*grouped_events + self.affine_beta
        return new_events


class Sample_Group(nn.Module):
    def __init__(self, nevent, radius, nsample, channel, normalize="center"):
        super(Sample_Group, self).__init__()
        self.nevent = nevent
        self.radius = radius
        self.nsample = nsample
        self.channel = channel
        self.normalize = normalize
        self.normalizer = Normalize(channel,self.normalize)


    def forward(self,xy,events):
        B, N, C = xy.shape
        # SES
        _,fps_idx = surface_event_sample(xy, self.nevent)
        # FPS
        # fps_idx = farthest_point_sample(xy, self.nevent)  # [B,nevent]
        # RS
        # fps_idx = random_sample(xy,self.nevent)
        torch.cuda.empty_cache()
        new_xy = index_events(xy, fps_idx)                # [B,nevent,2]
        torch.cuda.empty_cache()
        idx = research_circle(self.radius, self.nsample, xy, new_xy)  # [B,nevent,nsample]
        torch.cuda.empty_cache()
        grouped_xy = index_events(xy, idx)                # [B,nevent,nsample,2]
        torch.cuda.empty_cache()
        grouped_xy_norm = grouped_xy - new_xy.view(B, self.nevent, 1, C)  # new_xy表示centriods;这是求每
        torch.cuda.empty_cache()

        if events is not None:
            if self.normalize is not None:
                new_events = index_events(events, fps_idx)
                grouped_events = index_events(events, idx)
                torch.cuda.empty_cache()
                new_events = self.normalizer(new_xy, new_events, grouped_xy, grouped_events)
            else:
                grouped_events = index_events(events,idx)
                torch.cuda.empty_cache()
                new_events = torch.cat([grouped_events,grouped_xy_norm], dim=-1)   #[B,npoint,nsample,2+D]

        else:
            new_events = grouped_xy_norm

        return new_xy, new_events




def sample_and_group_all(xy, events):
    """
    输入: xy: input points position data, [B, N, 2]
         points: input points data, [B, N, D]
    输出: new_xy: sampled points position data, [B, 1, 2]
         new_events: sampled points data, [B, 1, N, 2+D]
    """
    device = xy.device
    B, N, C = xy.shape
    # Method-1
    new_xy = torch.zeros(B, 1, C).to(device)                                # 假设group all这种方式：1个点集只有1个centriod，[0,0]
    grouped_xy = xy.view(B, 1, N, C)
    # Method-2
    # new_xy = torch.mean(xy, dim=-2).view(B, 1, C)                            # 用质心替代[0,0]
    # centriods = new_xy.view(B, 1, 1, C)
    # grouped_xy = xy.view(B, 1, N, C) - centriods
    if events is not None:
        new_events = torch.cat([grouped_xy, events.view(B, 1, N, -1)], dim=-1)
    else:
        new_events = grouped_xy
    return new_xy, new_events



class AERNet_EFE(nn.Module):
    def __init__(self, nevent, radius, nsample, in_channel, mlp, normalize, group_all):
        super(AERNet_EFE, self).__init__()    # 我们在子类(AERNet_EFE)中需要调用父类(nn.Module)的方法时才这么用
        self.nevent = nevent
        self.radius = radius
        self.nsample = nsample
        self.in_channel = in_channel
        self.normalize = normalize
        self.sample_and_group = Sample_Group(self.nevent, self.radius, self.nsample, self.in_channel, self.normalize)

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all


    def forward(self,xy,events):
        if self.group_all:
            new_xy, new_events = sample_and_group_all(xy,events)
        else:
            new_xy, new_events = self.sample_and_group(xy,events)

        new_events = new_events.permute(0, 3, 2, 1)            # [B,2+D,nsample,npoint],便于卷积
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_events =  F.relu(bn(conv(new_events)))

        new_events = torch.max(new_events, 2)[0]
        new_events = new_events.permute(0, 2, 1)
        return new_xy, new_events


class get_model(nn.Module):
    def __init__(self,embedding_size):
        super(get_model, self).__init__()
        self.efe1_1 = AERNet_EFE(nevent=750, radius=0.1, nsample=160, in_channel=2, mlp=[32,64],normalize='center',group_all=False)
        # self.efe1_2 = AERNet_EFE(nevent=350, radius=0.16, nsample=140, in_channel=64+2, mlp=[64,96,128],normalize='center',group_all=False)
        self.efe1_3 = AERNet_EFE(None, None, None, in_channel=64+2, mlp=[64,128],normalize='center',group_all=True)                             # [1,512]
        self.efe2_1 = AERNet_EFE(None, None, None, in_channel=2, mlp=[32,64],normalize='center',group_all=True)                                            # [1,256]

        self.classifier = nn.Sequential(
            nn.Linear(128+64, embedding_size))


    def forward(self, xy):
        B, N, C = xy.shape
        xy = xy[:,:,:2]
        l1_xy, l1_events = self.efe1_1(xy, None)                # [B*16,2,250],[B,64,250]
        # l2_xy, l2_events = self.efe1_2(l1_xy, l1_events)        # [B*16,2,120],[B,128,120]
        l3_xy, l3_events = self.efe1_3(l1_xy, l1_events)        # [B*16,2,1],[B,256,1]
        l4_xy, l4_events = self.efe2_1(xy, None)                # [B*16,2,1],[B,96,1]
        concat_events = torch.cat([l3_events, l4_events], dim=-1)  # [B*16,1,352]
        x = concat_events.view(B,128+64)                         # [B,352]
        x = self.classifier(x)

        return x