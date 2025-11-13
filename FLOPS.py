import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from model.LITM import LITM
from model.UWCNN import UWCNN
from model.UTrans import UTrans
from model.NU2Net import NU2Net
from model.FIVE_APLUS import FIVE_APLUSNet

# 统一输入尺寸
dummy_input = torch.randn(1, 3, 256, 256).cuda()

# 模型列表：名称 + 构造函数
model_dict = {
    "UWCNN": UWCNN,
    "LITM": LITM,
    "UTrans": UTrans,
    "NU2Net": NU2Net,
    "FIVE_APLUSNet": FIVE_APLUSNet,
}

# 用于存储最终结果
results = []

for name, model_cls in model_dict.items():
    try:
        model = model_cls().cuda()
        model.eval()

        # 计算 FLOPs 和参数
        flops = FlopCountAnalysis(model, dummy_input)
        flops_val = flops.total() / 1e9  # Giga FLOPs

        param_info = parameter_count(model)
        total_params = sum(p for p in param_info.values()) / 1e6  # Mega Params

        results.append((name, total_params, flops_val))
    except Exception as e:
        results.append((name, "ERROR", str(e)))

# ====== 最后统一打印结果 ======
print("{:<15} {:>12} {:>15}".format("Model", "Params (M)", "FLOPs (G)"))
print("=" * 45)
for name, params, flops in results:
    if params == "ERROR":
        print(f"{name:<15} {'ERROR':>12} {flops:>15}")
    else:
        print("{:<15} {:>12.2f} {:>15.2f}".format(name, params, flops))
