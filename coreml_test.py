import coremltools
from PIL import Image
import numpy as np

__author__ = 'roeiherz'

# WEIGHT_PATH = 'outputs/UNET_224_weights_100000_days/0-best.pth'
model = coremltools.models.MLModel('outputs/UNET_224_weights_100000_days/0-best.mlmodel')
img = np.load('img.p')
img = img.transpose(1, 2, 0)
img = Image.fromarray(img, 'RGB')
# print(np.average(img[0, :, :]))
# print(np.average(img[1, :, :]))
# print(np.average(img[2, :, :]))

predictions = model.predict({'0': img})
# print(np.max(predictions['592'][0]))
# mask_img = Image.fromarray(np.uint8(predictions['590'][0] * 255), 'L')
mask_img = np.array(predictions['590']) * 255
mask_img = Image.fromarray(mask_img)
mask_img.show()

# img = Image.open('011934.jpg')
# img = img.resize((224,224),Image.BILINEAR)
# #img.show()
# img_array = np.array(img).astype('float32')
# print(np.average(img_array[0,:,:]))
# print(np.average(img_array[1,:,:]))
# print(np.average(img_array[2,:,:]))
#
# img_array = np.divide(img_array,255.0)
# #print(img_array)
#
# #img_pickle = cPickle.load('img.p')
# #img_pickle_PIL = Image.fromarray(img_pickle,'RGB')
#
# img = Image.fromarray(img_array,'RGB')
# print(np.average(img_array[0,:,:]))
# print(np.average(img_array[1,:,:]))
# print(np.average(img_array[2,:,:]))
#
#
#
# predictions = model.predict({'0':img})
# #print(np.max(predictions['592'][0]))
#
# mask_img = Image.fromarray(np.uint8(predictions['592'][0]*255),'L')
# #mask_img.show()
#
