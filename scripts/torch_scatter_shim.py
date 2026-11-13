"""Pure-torch stand-in for the three torch_scatter functions the repo imports (dim-0 use, differentiable)."""
import torch
def _shape(src, index, dim, dim_size):
    if index.dim() < src.dim():
        for _ in range(src.dim() - index.dim()): index = index.unsqueeze(-1)
    index = index.expand_as(src)
    n = int(index.max().item()) + 1 if dim_size is None else dim_size
    shp = list(src.shape); shp[dim] = n
    return index, shp
def scatter_sum(src, index, dim=0, out=None, dim_size=None):
    index, shp = _shape(src, index, dim, dim_size)
    o = torch.zeros(shp, dtype=src.dtype, device=src.device) if out is None else out
    return o.scatter_add(dim, index, src)
def scatter_mean(src, index, dim=0, out=None, dim_size=None):
    s = scatter_sum(src, index, dim, out, dim_size)
    c = scatter_sum(torch.ones_like(src), index, dim, None, s.shape[dim]).clamp(min=1)
    return s / c
def scatter_max(src, index, dim=0, out=None, dim_size=None):
    index, shp = _shape(src, index, dim, dim_size)
    o = torch.full(shp, float('-inf'), dtype=src.dtype, device=src.device)
    return o.scatter_reduce(dim, index, src, reduce='amax', include_self=True), None
def scatter_softmax(src, index, dim=0, eps=1e-12):
    mx = scatter_max(src, index, dim)[0]
    idx, _ = _shape(src, index, dim, mx.shape[dim])
    e = (src - mx.gather(dim, idx)).exp()
    return e / (scatter_sum(e, index, dim, None, mx.shape[dim]).gather(dim, idx) + eps)
