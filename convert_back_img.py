import cv2
import numpy as np
# img = cv2.imread('Visdrone/result/img_epoch_1_11150.png')
# img = img.astype(np.float64)
# map = cv2.imread('Visdrone/result/predict_epoch_1_11150.png')
# map = map.astype(np.float64)
# mean = (0.485, 0.456, 0.406)
# std = (0.229, 0.224, 0.225)
# img -= mean
# img /= std
# img /= 255.0
# img *= std
# img += mean
# img *=255.0
# output = img + map
# cv2.imwrite('1111.png', output)

# for i in range(19540):
#     img = cv2.imread('Visdrone/result/img_epoch_1_'+str(i)+'.png')
#     img = img.astype(np.float64)
#     map = cv2.imread('Visdrone/result/predict_epoch_1_'+str(i)+'.png')
#     map = map.astype(np.float64)
#     output = img + map
#     cv2.imwrite('Visdrone/combine_drone/output_epoch_1_'+str(i)+'.png', output)

img_array = []
for i in range(19540):
    img = cv2.imread('visdrone/combine_drone/output_epoch_1_'+str(i)+'.png')
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('video_15fps.avi', cv2.VideoWriter_fourcc('X','V','I','D'), 15, size)
for i in range(19540):
    print(i)
    out.write(img_array[i])
out.release()