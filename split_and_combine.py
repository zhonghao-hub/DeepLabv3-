import cv2
import numpy as np
import os

# images_base = 'visdrone/val/image(drone)/'
images_base = 'visdrone/DERT_resut_split/'
Path = [os.path.join(looproot, filename)
            for looproot, dirs, filenames in os.walk(images_base)
            for filename in filenames if filename.endswith('jpg')]
Path.sort(key=lambda x: x[-12:-4])
print(len(Path))

# for p in Path:
#     print(p)
#     img = cv2.imread(p)
#
#     # height, width, layers = img.shape
#     # size = (width, height)
#     img = cv2.resize(img, (3591, 2052))
#     for i in range(4):
#         for j in range(7):
#             temp = img[i*513:(i+1)*513,j*513:(j+1)*513]
#             # print(temp.shape)
#             cv2.imwrite('visdrone/Image_path_DETR/image(drone_split)/'+p[-15:-4]+str(i)+str(j)+'.jpg', temp)


for i in range(0, len(Path), 28):
    print(i)
    # temp = np.zeros((2052,3591, 3))
    temp = np.zeros((3200, 5600, 3))
    for m in range(4):
        for n in range(7):
            # temp[m * 513:(m + 1) * 513, n * 513:(n + 1) * 513] = cv2.imread(Path[i+m*7+n])
            temp[m * 800:(m + 1) * 800, n * 800:(n + 1) * 800] = cv2.imread(Path[i + m * 7 + n])
    cv2.imwrite('visdrone/combine_drone(parking_lot)/'+str(i//28)+'111.jpg', temp)


