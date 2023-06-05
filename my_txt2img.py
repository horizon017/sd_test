import os
import random

import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.util import instantiate_from_config

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def main():
    a = random.randint(1, 999999999)
    seed_everything(a)

    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, "models\ldm\stable-diffusion-v1\model.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)
    start_code = None

    #TODO:  [0]批次大小   [1]CFG缩放    [2]采样步数     [3]通道数    [4]高    [5]宽
    #TODO:  [6]下采样因子    [7]采样数量     [8]确定性因子    [9]迭代次数     [10]提示词
    d = [1, 9, 40, 4, 576, 576, 8, 1, 0.0, 1, "a cat"]
    batch_size, scale, ddim_steps, C, H, W, f, n_samples, ddim_eta, n_iter, prompt = d[0], *d[1:]

    outpath = "output/samples"
    grid_count = len(os.listdir(outpath)) - 1

    data = [batch_size * [prompt]]

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        print("embedding shape:", c.shape)
                        shape = [C, H // f, W // f]
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                         conditioning=c,
                                                         batch_size=n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc,
                                                         eta=ddim_eta,
                                                         x_T=start_code)
                        print("samples shape:", samples_ddim.shape)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().numpy()

                        print("image shape:", x_samples_ddim.shape)
                        x_sample = 255. * x_samples_ddim
                        img1 = np.stack([x_sample[0][0, :, :], x_sample[0][1, :, :], x_sample[0][2, :, :]], axis=2)
                        img = Image.fromarray(img1.astype(np.uint8))
                        img.save(os.path.join(outpath, f'{grid_count:04}-{str(a)}.png'))
                        grid_count += 1


if __name__ == "__main__":
    main()