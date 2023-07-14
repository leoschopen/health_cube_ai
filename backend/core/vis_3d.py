
import nibabel as nib
import matplotlib.pyplot as plt
from mayavi import mlab


nii_img = nib.load('../data/0b2be9e0-886b-4144-99f0-8bb4c6eaa848_0000.nii.gz')
img_data = nii_img.get_fdata()

# 使用 matplotlib 可视化一部分切片
# plt.imshow(img_data[:, :, 50], cmap='gray')
# plt.show()

# 使用 mayavi 可视化 3D 图像
mlab.figure(bgcolor=(1, 1, 1))
mlab.contour3d(img_data, contours=10, transparent=True)
mlab.show()