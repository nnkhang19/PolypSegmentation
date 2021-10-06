from attention import *
from unet_parts import *


if __name__ == '__main__':
	block = AttentionBlock(512)

	x = torch.randn(5, 512, 300, 300)

	out, map1, map2 = block(x)

	print(out.shape)
	print(map1.shape)
	print(map2.shape)