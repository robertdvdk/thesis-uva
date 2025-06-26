"""
Create a subset of ImageNet, based on two parameters: the proportion of images
per class, and the proportion of all classes to use. Splits this new subset
into train, val, and test.
"""


import numpy as np
import os
import shutil


def create_subset(proportion_imgs: float, proportion_cls: float, datasets_path: str, overwrite: bool=False) -> str:
    """
    Create a subset of ImageNet based on two parameters: the proportion of images
    per class, and the proportion of all classes to use.

    Args:
        proportion_imgs:
            The proportion of images to use per class.
        proportion_cls:
            The proportion of all ImageNet classes to use.
        datasets_path:
            The path to the dataset folder, containing the ImageNet directory.
        overwrite:
            Whether to overwrite existing dataset.

    Returns:
        Path to the new dataset.
    """
    imagenet_path = os.path.join(datasets_path, "imagenet/ILSVRC/Data/CLS-LOC/train")

    classes_in_dset = int(1000 * proportion_cls)
    subset_path = os.path.join(datasets_path, f"subset_imagenet_{classes_in_dset}-{proportion_imgs}")

    if os.path.exists(subset_path):
        if overwrite:
            shutil.rmtree(subset_path)
        else:
            return subset_path

    classes = os.listdir(imagenet_path)
    np.random.shuffle(classes)
    picked_classes = classes[:classes_in_dset]

    all_imgs = {}
    for cls in picked_classes:
        all_imgs[cls] = os.listdir(os.path.join(imagenet_path, cls))

    picked_imgs = {}
    for cls, imgs in all_imgs.items():
        np.random.shuffle(imgs)
        picked_imgs[cls] = imgs[:int(len(imgs) * proportion_imgs)]

    if not os.path.exists(subset_path):
        os.makedirs(subset_path)

    for cls, imgs in picked_imgs.items():
        print(f"Writing {len(imgs)} images to {subset_path}/{cls}")
        if not os.path.exists(os.path.join(subset_path, cls)):
            os.makedirs(os.path.join(subset_path, cls))
        for img in imgs:
            shutil.copyfile(os.path.join(imagenet_path, cls, img), os.path.join(subset_path, cls, img))

    return 1


def split_train_val_test(train_ratio: float, val_ratio: float, test_ratio: float, subset_path: str) -> None:
    """
    Split the newly created subset into train, val, and test.

    Args:
        train_ratio:
            The proportion of images to train on.
        val_ratio:
            The proportion of images to validate on.
        test_ratio:
            The proportion of images to test on.
        subset_path:
            Path to the dataset.
    """
    assert train_ratio + val_ratio + test_ratio == 1
    train_path = os.path.join(subset_path, "train")
    val_path = os.path.join(subset_path, "val")
    test_path = os.path.join(subset_path, "test")

    if not os.path.exists(train_path):
        os.makedirs(train_path)
        os.makedirs(val_path)
        os.makedirs(test_path)
    else:
        return

    for cls in os.listdir(subset_path):
        if cls in ["train", "val", "test"]:
            continue
        os.makedirs(os.path.join(train_path, cls))
        os.makedirs(os.path.join(val_path, cls))
        os.makedirs(os.path.join(test_path, cls))
        imgs = os.listdir(os.path.join(subset_path, cls))
        np.random.shuffle(imgs)
        train_imgs = imgs[:int(len(imgs) * train_ratio)]
        val_imgs = imgs[int(len(imgs) * train_ratio):int(len(imgs) * (train_ratio + val_ratio))]
        test_imgs = imgs[int(len(imgs) * (train_ratio + val_ratio)):]

        for img in train_imgs:
            shutil.move(os.path.join(subset_path, cls, img), os.path.join(train_path, cls, img))

        for img in val_imgs:
            shutil.move(os.path.join(subset_path, cls, img), os.path.join(val_path, cls, img))

        for img in test_imgs:
            shutil.move(os.path.join(subset_path, cls, img), os.path.join(test_path, cls, img))

        shutil.rmtree(os.path.join(subset_path, cls))


def main() -> None:
    np.random.seed(0)
    subset_path = create_subset(proportion_imgs=0.03,
                  proportion_cls=1,
                  datasets_path="/home/robert/Projects/data",
                  overwrite=False)

    split_train_val_test(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, subset_path=subset_path)

if __name__ == "__main__":
    main()