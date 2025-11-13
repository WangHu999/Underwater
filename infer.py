
# 计算平均每张图片的推理时间
import os
import torch
import argparse
import time
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.UIEB import UIEBDataset
from model.UWCNN import UWCNN
from model.LITM import LITM
from model.NU2Net import NU2Net
from model.FIVE_APLUS import FIVE_APLUSNet
from model.UTrans import UTrans

from myutils.quality_refer import calc_psnr, calc_mse, calc_ssim, normalize_img

parser = argparse.ArgumentParser(description='Testing UIEB dataset')
parser.add_argument('--model_name', type=str, default='LITM',
                    help='model name, options:[LITM, UTrans, NU2Net, UWCNN, FIVE_APLUS]')
parser.add_argument('--crop_size', type=int, default=256, help='crop size')
parser.add_argument('--input_norm', action='store_true', help='norm the input image to [-1,1]')
hparams = parser.parse_args()

model_path = './checkpoints/UIEB5/' + hparams.model_name + '.ckpt'

pred_path = './data/UIEB/All_Results/' + hparams.model_name + '/U90/'
if not os.path.exists(pred_path):
    os.makedirs(pred_path)

pred_set = UIEBDataset("./data/", train_flag=False, pred_flag=True,
                       train_size=hparams.crop_size, input_norm=hparams.input_norm)
pred_loader = DataLoader(pred_set, batch_size=1, shuffle=False)

model_zoos = {
    "UWCNN": UWCNN,
    "LITM": LITM,
    "UTrans": UTrans,
    "NU2Net": NU2Net,
    "FIVE_APLUS": FIVE_APLUSNet,
}
model = model_zoos[hparams.model_name]().cuda()
ckpt = torch.load(model_path)
ckpt = ckpt['state_dict']
new_ckpt = {key[6:]: value for key, value in ckpt.items()}
missing_keys, unexpected_keys = model.load_state_dict(new_ckpt, strict=False)
print("missing keys: ", missing_keys)
print("unexpected keys: ", unexpected_keys)
model.eval()

total_time = 0.0
num_images = 0

print("Generate enhanced images for challenging set and calculate inference time...")
for idx, (x, y, filename) in tqdm(enumerate(pred_loader), total=len(pred_loader)):
    with torch.no_grad():
        x = x.cuda()
        start_time = time.time()      # 开始计时
        y_hat = model(x)
        torch.cuda.synchronize()      # 保证 GPU 上的操作全部结束
        end_time = time.time()        # 结束计时
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        num_images += 1

        pred_img_tensor = normalize_img(y_hat)
        save_image(pred_img_tensor[0], os.path.join(pred_path, filename[0]), normalize=False)

average_time = total_time / num_images if num_images > 0 else 0
print(f"Average inference time per image: {average_time:.4f} seconds")
