# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader


class INatDataset(ImageFolder):
    def __init__(
        self,
        root,
        train=True,
        year=2018,
        transform=None,
        target_transform=None,
        category="name",
        loader=default_loader,
    ):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, "categories.json")) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter["annotations"]:
            king = []
            king.append(data_catg[int(elem["category_id"])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data["images"]:
            cut = elem["file_name"].split("/")
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


# OLD VERSION
# class ADE20KSegmentation(Dataset):
#     def __init__(self, root, is_train=True, image_size=224):

#         super().__init__()
#         if is_train:
#             split = "training"
#         else:
#             split = "validation"
#         self.image_files = sorted(
#             glob.glob(os.path.join(root, split, "**", "*.jpg"), recursive=True)
#         )
#         self.image_size = image_size
#         self.transform = transforms.Compose(
#             [
#                 transforms.Resize((image_size, image_size)),
#                 transforms.ToTensor(),
#             ]
#         )

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_path = self.image_files[idx]
#         mask_path = img_path.replace(".jpg", "_seg.png")

#         image = Image.open(img_path).convert("RGB")
#         mask = Image.open(mask_path).convert("L")

#         image = self.transform(image)
#         mask = np.array(mask)
#         mask = Image.fromarray(mask).resize(
#             (self.image_size, self.image_size), resample=Image.NEAREST
#         )
#         mask = np.array(mask).astype(np.int64)

#         mask[(mask != 255) & (mask > 149)] = 149  # TODO: clarify
#         mask = torch.from_numpy(mask)

#         return image, mask


# NEW VERSION
class ADE20KSegmentation(Dataset):

    def __init__(self, root: str, is_train: bool = True, image_size: int = 224):

        self.root = root
        self.mode = "train" if is_train else "valid"
        self.image_size = image_size

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

        self.get_filenames()

    def get_filenames(self):
        self.images_training_dir = os.path.join(self.root, "images", "training")
        self.annotations_training_dir = os.path.join(
            self.root, "annotations", "training"
        )

        self.images_validation_dir = os.path.join(self.root, "images", "validation")
        self.annotations_validation_dir = os.path.join(
            self.root, "annotations", "validation"
        )

        train_images = sorted(
            [f for f in os.listdir(self.images_training_dir) if f.endswith(".jpg")]
        )
        train_masks = sorted(
            [f for f in os.listdir(self.annotations_training_dir) if f.endswith(".png")]
        )

        val_images = sorted(
            [f for f in os.listdir(self.images_validation_dir) if f.endswith(".jpg")]
        )
        val_masks = sorted(
            [
                f
                for f in os.listdir(self.annotations_validation_dir)
                if f.endswith(".png")
            ]
        )

        if self.mode == "train":
            self.images, self.masks = train_images, train_masks
            self.images_path, self.masks_path = (
                self.images_training_dir,
                self.annotations_training_dir,
            )
        else:
            self.images, self.masks = val_images, val_masks
            self.images_path, self.masks_path = (
                self.images_validation_dir,
                self.annotations_validation_dir,
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx].split("/")[-1].split(".")[0]

        image_path = os.path.join(self.images_path, f"{image_name}.jpg")
        mask_path = os.path.join(self.masks_path, f"{image_name}.png")

        # Load image and apply transform
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        # Load and resize mask without normalization
        mask = Image.open(mask_path)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()  # shape: [H, W]
        return image, mask

        # return {
        #     "image": image,
        #     "mask": mask,
        #     "image_filepath": image_path,
        #     "mask_filepath": mask_path,
        # }


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == "CIFAR":
        try:
            dataset = datasets.CIFAR100(
                args.data_path, train=is_train, transform=transform
            )
        # download it if needed
        except RuntimeError:
            dataset = datasets.CIFAR100(
                args.data_path, train=is_train, transform=transform, download=True
            )
        nb_classes = 100
    elif args.data_set == "IMNET":
        root = os.path.join(args.data_path, "train" if is_train else "val")
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "INAT":
        dataset = INatDataset(
            args.data_path,
            train=is_train,
            year=2018,
            category=args.inat_category,
            transform=transform,
        )
        nb_classes = dataset.nb_classes
    elif args.data_set == "INAT19":
        dataset = INatDataset(
            args.data_path,
            train=is_train,
            year=2019,
            category=args.inat_category,
            transform=transform,
        )
        nb_classes = dataset.nb_classes
    elif args.data_set == "ADE20K":
        dataset = ADE20KSegmentation(args.data_path, is_train=is_train)
        nb_classes = args.segmentation_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(
                size, interpolation=3
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
