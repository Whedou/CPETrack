import os
import sys
import time
import torch
import importlib
from thop import profile
from thop.utils import clever_format

prj_path = os.path.join(os.getcwd(), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

config_module = importlib.import_module('lib.config.cpetrack.config')
cfg = config_module.cfg
config_module.update_config_from_file(os.path.join(prj_path, 'experiments/cpetrack/deep_rgbt_256.yaml'))

model_module = importlib.import_module('lib.models')
model = model_module.build_cpetrack(cfg, training=False).cuda().eval()

bs = 1
z_sz = cfg.TEST.TEMPLATE_SIZE
x_sz = cfg.TEST.SEARCH_SIZE

template = [
    torch.randn(bs, 6, z_sz, z_sz).cuda()
    for _ in range(cfg.DATA.TEMPLATE.NUMBER)
]
search = [
    torch.randn(bs, 6, x_sz, x_sz).cuda()
]

depth = 12
keep_rate = [x for x in torch.linspace(0.7, 1, depth // 4)][::-1]

class Wrapper(torch.nn.Module):
    def __init__(self, model, keep_rate):
        super().__init__()
        self.model = model
        self.keep_rate = keep_rate

    def forward(self, template, search):
        return self.model(
            template=template,
            search=search,
            keep_rate=self.keep_rate,
            training=False,
            tgt_pre=[]
        )

wrapped = Wrapper(model, keep_rate).cuda().eval()

macs, params = profile(wrapped, inputs=(template, search), verbose=False)
macs_fmt, params_fmt = clever_format([macs, params], "%.3f")
print("overall macs:", macs_fmt)
print("overall params:", params_fmt)

T_w, T_t = 100, 1000
with torch.no_grad():
    torch.cuda.synchronize()
    for _ in range(T_w):
        _ = wrapped(template, search)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(T_t):
        _ = wrapped(template, search)
    torch.cuda.synchronize()
    end = time.time()

avg_lat = (end - start) / T_t
print("avg latency(ms): %.2f" % (avg_lat * 1000))
print("FPS: %.2f" % (1.0 / avg_lat))
