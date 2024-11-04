import torch
import matplotlib.pyplot as plt
import cv2 as cv
origin = cv.imread('C:/Users/samsung/OneDrive/Desktop/taco ex/stylegan2-ada-pytorch/origin.png')
advimg = cv.imread('C:/Users/samsung/OneDrive/Desktop/taco ex/stylegan2-ada-pytorch/imgal1.png')
advimg = cv.resize(advimg, (origin.shape[1], origin.shape[0]))  
origin = cv.cvtColor(origin, cv.COLOR_BGR2RGB)
advimg = cv.cvtColor(advimg, cv.COLOR_BGR2RGB)
origin_tensor = torch.from_numpy(origin).float() / 255.0
advimg_tensor = torch.from_numpy(advimg).float() / 255.0
difference = torch.abs(origin_tensor - advimg_tensor)
difference = difference / difference.max()
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(origin_tensor.numpy())
plt.subplot(1, 3, 2)
plt.title("Adversarial Image")
plt.imshow(advimg_tensor.numpy())
plt.subplot(1, 3, 3)
plt.title("Difference")
plt.imshow(difference.numpy())
plt.show()
