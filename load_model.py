import os

import src.models.vision_transformer as vit
from src.utils.tensors import trunc_normal_

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

def make_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):
    transform_list = []
    transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    transform = transforms.Compose(transform_list)
    return transform


def make_videoframes(
    batch_size,
    transform=None,
    collator=None,
    pin_mem=True,
    num_workers=1,
    root_path=None,
    image_folder=None,
    drop_last=True,
):
    dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=transform
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)

    return dataset, data_loader


class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root,
        image_folder,
        transform=None,
        suffix='train',
    ):
        """
        ImageNet

        Dataset wrapper (can copy data locally to machine)

        :param root: root network directory for ImageNet data
        :param image_folder: path to images inside root network directory
        :param suffix: 'train/' if train else 'val/'
        """
        data_path = os.path.join(root, image_folder, suffix)
        super(ImageNet, self).__init__(root=data_path, transform=transform)


def init_model(
    device,
    patch_size=14,
    model_name='vit_small',
    crop_size=224,
    pred_depth=12,
    pred_emb_dim=384
):
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size)
    predictor = vit.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    return encoder, predictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
encoder, predictor = init_model(device=device)

save_dir = '/scratch/tw2672/1008_Project'
checkpoint_path = 'jepa-latest.pth.tar'
checkpoint = torch.load(os.path.join(save_dir, checkpoint_path))

# Remove the "module." prefix from keys
pretrained_dict = checkpoint['encoder']
new_pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
encoder.load_state_dict(new_pretrained_dict, strict=False)
encoder.eval()

class ImageDecoder(nn.Module):
    def __init__(self, embedding_dim):
        super(ImageDecoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 224*224*3),
            nn.Tanh()  # You can use Tanh activation to constrain output to [-1, 1]
        )

    def forward(self, x):
        x = x.view(-1, self.embedding_dim)
        x = self.fc(x)
        return x.view(-1, 3, 224, 224)

transform = make_transforms()
_, dataloader = make_videoframes(batch_size=32, transform=transform, root_path='/scratch/tw2672/1008_Project', image_folder='data')

embedding_dim = 256*384
decoder = ImageDecoder(embedding_dim).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(decoder.parameters(), lr=0.001)

# train
# best_loss = np.inf
# for epoch in range(100):
#     total_loss = 0
#     for itr, udata in enumerate(dataloader):
#         imgs = udata[0].to(device, non_blocking=True)
#         with torch.no_grad():
#             z = encoder(imgs).detach()
#         reconstructed_imgs = decoder(z)
#         optimizer.zero_grad()
#         loss = criterion(reconstructed_imgs, imgs)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     if total_loss < best_loss:
#         torch.save(decoder.state_dict(), 'decoder.pth.tar')

# reconstruct
checkpoint = torch.load('decoder.pth.tar')
decoder.load_state_dict(checkpoint)
for iter, udata in enumerate(dataloader):
    imgs = udata[0].to(device, non_blocking=True)
    with torch.no_grad():
        z = encoder(imgs).detach()
    reconstructed_imgs = decoder(z)
    break

batch_size = reconstructed_imgs.size(0)

for i in range(batch_size):
    img = reconstructed_imgs[i].detach().cpu().permute(1,2,0)
    plt.imshow(img.numpy())
    plt.axis('off')
    plt.savefig(f'images/image_{i}.png', bbox_inches='tight', pad_inches=0, dpi=300)