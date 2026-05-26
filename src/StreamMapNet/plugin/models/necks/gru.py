import torch
import torch.nn as nn
from mmdet.models import NECKS
from mmcv.cnn.utils import kaiming_init, constant_init


@NECKS.register_module()
class ConvGRU(nn.Module):
    def __init__(self, out_channels):
        super(ConvGRU, self).__init__()
        kernel_size = 1
        padding = kernel_size // 2
        self.convz = nn.Conv2d(2*out_channels, 
            out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.convr = nn.Conv2d(2*out_channels, 
            out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.convq = nn.Conv2d(2*out_channels, 
            out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.ln = nn.LayerNorm(out_channels)
        self.debug_cfg = {}
        self.last_debug = None

    def set_debug_cfg(self, debug_cfg=None):
        self.debug_cfg = debug_cfg or {}

    def _debug_enabled(self):
        return self.debug_cfg.get('enabled', False)

    def _stats(self, tensor):
        flat = tensor.detach().float().reshape(-1)
        quantiles = torch.quantile(
            flat, flat.new_tensor([0.10, 0.50, 0.90, 0.95]))
        return {
            'mean': flat.mean(),
            'p10': quantiles[0],
            'p50': quantiles[1],
            'p90': quantiles[2],
            'p95': quantiles[3],
        }

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, h, x):
        if len(h.shape) == 3:
            h = h.unsqueeze(0)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        hx = torch.cat([h, x], dim=1) # [1, 2c, h, w]
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        new_x = torch.cat([r * h, x], dim=1) # [1, 2c, h, w]
        q = self.convq(new_x)

        out = ((1 - z) * h + z * q).squeeze(0) # (1, C, H, W)
        out = self.ln(out.permute(1, 2, 0)).permute(2, 0, 1).contiguous()
        if self._debug_enabled():
            z_stats = self._stats(z)
            self.last_debug = {
                'z_mean': z_stats['mean'],
                'z_p10': z_stats['p10'],
                'z_p50': z_stats['p50'],
                'z_p90': z_stats['p90'],
                'z_p95': z_stats['p95'],
                'history_bev_norm': h.detach().float().norm(),
                'current_bev_norm': x.detach().float().norm(),
                'fused_bev_norm': out.detach().float().norm(),
            }
            if self.debug_cfg.get('save_full', False):
                self.last_debug['z'] = z.detach()
        else:
            self.last_debug = None
        return out
