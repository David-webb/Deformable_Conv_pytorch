import torch
from torch import nn

def _get_p_n(N=9, dtype=torch.int64):
    # 卷积核中每个点相对于卷积核中心点的偏移
    p_n_x, p_n_y = torch.meshgrid(
        torch.arange(-1, 2),
        torch.arange(-1,2))
    print( p_n_x, p_n_y)        
    # (2N, 1)
    p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
    p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

    print(p_n)
    return p_n

def _get_p_0(h=4, w=4, N=9, dtype=torch.int64):
    # 卷积核在特征图上滑动的中心点
    p_0_x, p_0_y = torch.meshgrid(
        torch.arange(1, h*1+1, 1),
        torch.arange(1, w*1+1, 1))
    # print(p_0_x.size(), p_0_y)
    p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
    p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
    p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
    # print(p_0)
    print(p_0.size())
    return p_0

def _get_p(self, offset, dtype=torch.int64):
    N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

    # (1, 2N, 1, 1)
    p_n = self._get_p_n(N, dtype)
    # (1, 2N, h, w)
    p_0 = self._get_p_0(h, w, N, dtype)
    p = p_0 + p_n + offset
    return p

if __name__ == "__main__":
    pn = _get_p_n()
    p0 = _get_p_0()
    print((p0+pn).size())
    pass
