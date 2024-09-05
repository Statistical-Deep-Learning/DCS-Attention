from torchvision import datasets, transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
    transforms.ColorJitter(brightness=jitter_param,
                           contrast=jitter_param,
                           saturation=jitter_param),
    Lighting(lighting_param),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
    ]))
val_dataset = datasets.ImageFolder(
    valdir, transforms.Compose([
        transforms.Resize(int(input_size / 0.875)),
        transforms.CenterCrop(input_size),
    ]))

if is_train:
    transforms = []
    if config.augmentation.use_random_crop:
        transforms.append(RandomResizeCrop(config))
    else:
        transforms.append(CenterCrop(config))
    if config.augmentation.use_random_horizontal_flip:
        transforms.append(RandomHorizontalFlip(config))

    transforms.append(Normalize(mean, std))

    if config.augmentation.use_cutout:
        transforms.append(Cutout(config))
    if config.augmentation.use_random_erasing:
        transforms.append(RandomErasing(config))
    if config.augmentation.use_dual_cutout:
        transforms.append(DualCutout(config))

    transforms.append(ToTensor())
else:
    transforms = []
    if config.tta.use_resize:
        transforms.append(Resize(config))
    if config.tta.use_center_crop:
        transforms.append(CenterCrop(config))
    transforms += [
        Normalize(mean, std),
        ToTensor(),
    ]