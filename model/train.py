"""
Training for CycleGAN

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-05: Initial coding
* 2022-12-21: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
from dataset import rgbnirDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def train_fn(
    disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    rgb_reals = 0
    rgb_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (nir, rgb) in enumerate(loop):
        nir = nir.to(config.DEVICE)
        rgb = rgb.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_rgb = gen_H(nir)
            D_rgb_real = disc_H(rgb)
            D_rgb_fake = disc_H(fake_rgb.detach())
            rgb_reals += D_rgb_real.mean().item()
            rgb_fakes += D_rgb_fake.mean().item()
            D_rgb_real_loss = mse(D_rgb_real, torch.ones_like(D_rgb_real))
            D_rgb_fake_loss = mse(D_rgb_fake, torch.zeros_like(D_rgb_fake))
            D_rgb_loss = D_rgb_real_loss + D_rgb_fake_loss

            fake_nir = gen_Z(rgb)
            D_nir_real = disc_Z(nir)
            D_nir_fake = disc_Z(fake_nir.detach())
            D_nir_real_loss = mse(D_nir_real, torch.ones_like(D_nir_real))
            D_nir_fake_loss = mse(D_nir_fake, torch.zeros_like(D_nir_fake))
            D_nir_loss = D_nir_real_loss + D_nir_fake_loss

            # put it togethor
            D_loss = (D_rgb_loss + D_nir_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators rgb and nir
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_rgb_fake = disc_H(fake_rgb)
            D_nir_fake = disc_Z(fake_nir)
            loss_G_H = mse(D_rgb_fake, torch.ones_like(D_rgb_fake))
            loss_G_Z = mse(D_nir_fake, torch.ones_like(D_nir_fake))

            # cycle loss
            cycle_nir = gen_Z(fake_rgb)
            cycle_rgb = gen_H(fake_nir)
            cycle_nir_loss = l1(nir, cycle_nir)
            cycle_rgb_loss = l1(rgb, cycle_rgb)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_nir = gen_Z(nir)
            identity_rgb = gen_H(rgb)
            identity_nir_loss = l1(nir, identity_nir)
            identity_rgb_loss = l1(rgb, identity_rgb)

            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_nir_loss * config.LAMBDA_CYCLE
                + cycle_rgb_loss * config.LAMBDA_CYCLE
                + identity_rgb_loss * config.LAMBDA_IDENTITY
                + identity_nir_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_rgb * 0.5 + 0.5, f"saved_images/rgb_{idx}.png")
            save_image(fake_nir * 0.5 + 0.5, f"saved_images/nir_{idx}.png")

        loop.set_postfix(rgb_real=rgb_reals / (idx + 1), rgb_fake=rgb_fakes / (idx + 1))


def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=1).to(config.DEVICE)
    gen_Z = Generator(in_channels=3,out_channels=1, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(in_channels=1,out_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H,
            gen_H,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z,
            gen_Z,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H,
            disc_H,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z,
            disc_Z,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = rgbnirDataset(
        root_rgb=config.TRAIN_DIR + "/rgbs",
        root_nir=config.TRAIN_DIR + "/nirs",
        transform=config.transforms,
    )
    val_dataset = rgbnirDataset(
        root_rgb="cyclegan_test/rgb1",
        root_nir="cyclegan_test/nir1",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_H,
            disc_Z,
            gen_Z,
            gen_H,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)


if __name__ == "__main__":
    main()
