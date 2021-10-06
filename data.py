import os 
import torch 
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image
import pandas as pd

class PolypGAN(Dataset):
	def __init__(self, csv_file, image_dir, mask_dir, transforms = None):
		self.image_dir = image_dir 
		self.transforms = transforms 
		self.mask_dir = mask_dir 
		self.filenames = pd.read_csv(csv_file)['file_name']

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		image_path = os.path.join(self.image_dir, self.filenames[idx])
		mask_path = os.path.join(self.mask_dir, self.filenames[idx])

		image = np.array(Image.open(image_path).convert("RGB"))
		mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.float32)

		if self.transforms is not None:
			trans = self.transforms(image = image)
			image = trans['image']
			m1 = self.transforms(image = mask)
			mask = m1['image']

		return image, mask 


class Polyp(Dataset):
	def __init__(self, csv_file, image_dir, mask_dir, image_transform = None, mask_transform = None):
		self.image_dir = image_dir 
		self.mask_dir = mask_dir
		self.filenames = pd.read_csv(csv_file)['file_name']

		self.image_transform = image_transform
		self.mask_transform = mask_transform

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, index):
		image = self._read_image(index)
		image = self._transform_image(image)
		mask = self._read_mask(index)
		mask = self._transform_mask(mask)
		return image, mask 

	def _read_image(self, index):
		image_path = os.path.join(self.image_dir, self.filenames[index])
		return np.array(Image.open(image_path).convert('RGB'))

	def _transform_image(self, image):
		if self.image_transform is not None:
			transform = self.image_transform(image = image)
			image = transform['image']
		return image 

	def _read_mask(self, index):
		mask_path = os.path.join(self.mask_dir, self.filenames[index])
		mask = np.array(Image.open(mask_path).convert('L'))
		return mask / 255.0

	def _transform_mask(self, mask):
		if self.mask_transform is not None:
			transform = self.mask_transform(image = mask)
			mask = transform['image']
		return mask

def show_image(image):
	plt.imshow(image)
	plt.show()

def test():
	transforms = A.Compose(
		[
			A.Resize(height=256, width=256),
			A.Normalize(
				mean= [0.485, 0.456, 0.406], 
				std = [0.229, 0.224, 0.225],
				max_pixel_value=255.0,
			),
			ToTensorV2(),
		],
	)
	transforms2 = A.Compose(
		[
			A.Resize(height=256, width=256),
			ToTensorV2(),
		]
	)
	image_dir = '/home/nhatkhang/Documents/MediaEval2021/images'
	mask_dir = '/home/nhatkhang/Documents/MediaEval2021/masks'
	csv_file = 'train.csv'

	dataset = Polyp(csv_file, image_dir, mask_dir, image_transform= transforms, mask_transform = transforms2)

	image, mask = dataset[1]

	print(mask.shape)
	print(mask.dtype)
	print(torch.max(mask))
	print(torch.min(mask))

if __name__ == '__main__':
	test()



