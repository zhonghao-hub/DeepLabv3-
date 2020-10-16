import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

import cv2

class VisdroneSegmentation(data.Dataset):
    NUM_CLASSES = 21

    def __init__(self, args, root=Path.db_root_dir('visdrone'), split="train"):

        self.root = root#'cityscapes'图片集的路径
        self.split = split#'train'
        self.args = args
        self.files = {}

        self.images_base = os.path.join(self.root, self.split, 'image')#images directory('peas/train/image/')
        self.annotations_base = os.path.join(self.root, self.split, 'label')#annotations directory ('peas/train/label/')

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.jpg')#self.files[split] 存的是image directory 中的图片list
        self.valid_classes = range(21)
        # self.void_classes = [1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        # self.valid_classes = [0, 15, 15, 2, 7, 5, 6, 11, 8, 9, 10, 0]
        # self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        # self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        # self.class_names = ['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']
        #
        # self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))#创建一个字典，将self.valid_classes 的数字map 到[0,1]

        if not self.files[split]:#若image directory 中不存在图片
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))# eg: 输出'Found 200 train images'

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):#返回transformation（Augmentation）之后的第index的（image，label）

        img_path = self.files[self.split][index].rstrip()
        # print(img_path)
        lbl_path = os.path.join(self.annotations_base,
                                os.path.basename(img_path)[:-4] + '.jpg')
        # print(lbl_path)


        # img_path = os.path.join(self.images_base, str(index) + '.jpg')#返回第'index'个图片的路径''peas/train/image/index.png'# ； rstrip() 删除 string 字符串末尾的指定字符（默认为空格)
        # lbl_path = os.path.join(self.annotations_base, str(index) + '.jpg')
        #'peas/train/label/index.png'

        _img = Image.open(img_path).convert('RGB')#-img 是将第index的图片转换成RGB格式\

        # # _lb = Image.open(lbl_path).convert('L')  # -img 是将第index的图片转换成RGB格式\
        # _tmp = np.array(Image.open(lbl_path).convert('L'), dtype=np.float32)  # -tmp 是将第index的图片对应的label转换成array
        # # print(_tmp)
        # _tm = self.encode_segmap(_tmp)  # 对于_tmp这个label图片，若是void_class则变为0，若是valid——class则变成相应map的值0～18
        # _lb = Image.fromarray(_tm)  # 将_tmp的array格式变回图片格式
        # sample = {'image': _img, 'label': _lb}#_img:第index的图片， _target：第index的label图片

        sample = {'image': _img}


        if self.split == 'train':
            return self.transform_tr(sample)#进行random Augmentation
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        Path = [os.path.join(looproot, filename)
            for looproot, dirs, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]
        Path.sort(key=lambda x: x[3:-4])
        return Path

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([#将各种transformation组合在一起
            # tr.RandomHorizontalFlip(),
            # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            # tr.RandomGaussianBlur(),
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 512
    args.crop_size = 512

    cityscapes_train = VisdroneSegmentation(args, split='train')

    dataloader= DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='cityscapes')#Decode segmentation class labels into a color image
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)

