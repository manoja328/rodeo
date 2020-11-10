class VOCDetection_flip(torchvision.datasets.VOCDetection):
    def __init__(self, img_folder, year, image_set, transforms):
        super(VOCDetection_flip, self).__init__(img_folder,  year, image_set)
        self._transforms = transforms

    def __getitem__(self, idx):
        real_idx = idx//2
        img, target = super(VOCDetection_flip, self).__getitem__(real_idx)
        target = dict(image_id=real_idx, annotations=target['annotation'])
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            # img = img[[2, 1, 0],:]

            if (idx % 2) == 0:
                height, width = img.shape[-2:]
                img = img.flip(-1)
                bbox = target["boxes"]
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox

        return img, target

    def __len__(self):
        return 2*len(self.images)