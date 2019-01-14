import cv2
import os
import numpy as np
import av
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataset import MaskDataset, get_img_files, get_img_files_eval
from nets.MobileNetV2_unet import MobileNetV2_unet

__author__ = 'roeiherz'

FILE_EXISTS_ERROR = (17, 'File exists')
N_CV = 5
IMG_SIZE = 224
RANDOM_STATE = 1
FPS = 5


def get_data_loaders(val_files):
    val_transform = Compose([
        Resize((IMG_SIZE, IMG_SIZE)),
        ToTensor(),
    ])

    val_loader = DataLoader(MaskDataset(val_files, val_transform),
                            batch_size=1,
                            shuffle=TabError,
                            pin_memory=True,
                            num_workers=4)
    return val_loader


def create_folder(path):
    """
    Checks if the path exists, if not creates it.
    :param path: A valid path that might not exist
    :return: An indication if the folder was created
    """
    folder_missing = not os.path.exists(path)

    if folder_missing:
        # Using makedirs since the path hierarchy might not fully exist.
        try:
            os.makedirs(path)
        except OSError as e:
            if (e.errno, e.strerror) == FILE_EXISTS_ERROR:
                print(e)
            else:
                raise

        print('Created folder {0}'.format(path))

    return folder_missing


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def video_to_frames(input_video, out_dir, refinment=1, fps=1):
    """

    :param input_video: path for input video
    :param out_dir: output path directory
    :param refinement:
    :param fps:
            1: default fps
            -1: automatic default depends differently per video
            any other integer
    :return:
    """
    video = av.open(input_video)
    rotation = int(video.streams[0].metadata.get('rotate', 0))
    vidcap = cv2.VideoCapture(input_video)

    # Jump using the fps inputs
    if fps == -1:
        duration = float(video.streams[0].duration * video.streams[0].time_base)
        frames = video.streams[0].frames
        fps = int(round(frames / duration))

    count = 0
    image_files = []
    counter = 0
    index = 0
    while True:
        success, image = vidcap.read()
        if not success:
            print("Finished/Error in video: {}".format(input_video))
            break
        counter += 1
        if ((counter - 1) % refinment) > 0:
            continue

        image = rotate_bound(image, rotation)
        outpath = os.path.join(out_dir, "%.6d.jpg" % (index))

        if count % fps == 0:
            cv2.imwrite(outpath, image)
            image_files.append(outpath)
            index += 1
        count = count + 1


def images_to_video(outvid_path, input_folder):
    """
    Create video from images
    :param outvid_path: output path
    :param input_folder: 
    :return: 
    """
    outvid = cv2.VideoWriter(outvid_path, cv2.VideoWriter_fourcc(*'MJPG'), 5.0, (224, 224))

    for i in range(1, 1000):
        if os.path.isfile(os.path.join(input_folder, 'frame' + str(i) + '.jpg')):
            I = cv2.imread(os.path.join(input_folder, 'frame' + str(i) + '.jpg'))
            outvid.write(I)

    outvid.release()

    return


if __name__ == '__main__':
    # input_video = "/home/roei/Datasets/Accidents1K/Videos/0d1f5146-858f-48a5-8c9a-47b87fc8b6a8.mov"
    input_video = "/home/roei/Downloads/incident-865ba5029fb5fefaae91b3e1e354f403.mp4"
    output_video = "/home/roei/mobile-semantic-segmentation/outputs/"
    model_path = "/home/roei/mobile-semantic-segmentation/outputs/UNET_224_weights_100000_days/0-best.pth"
    uuid = os.path.basename(input_video).split('.')[0]
    output_path = os.path.join(output_video, "{}_masked".format(os.path.basename(input_video).split('.')[0]))
    output_shape = (720, 1280)

    # Creates frames if they don't exists
    if not os.path.exists(output_path):
        create_folder(output_path)

    # Process the network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # data_loader = get_data_loaders(frames)
    model = MobileNetV2_unet(mode="eval")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    transform = Compose([Resize((IMG_SIZE, IMG_SIZE)), ToTensor()])

    # # Process the Video
    video = av.open(input_video)
    rotation = int(video.streams[0].metadata.get('rotate', 0))
    # Video Reader
    vidcap = cv2.VideoCapture(input_video)

    # Jump using the fps inputs
    fps = FPS
    if fps == -1:
        duration = float(video.streams[0].duration * video.streams[0].time_base)
        frames = video.streams[0].frames
        fps = int(round(frames / duration))

    # Video Writer
    outvid = cv2.VideoWriter(os.path.join(output_path, "{}.avi".format(uuid)),
                             cv2.VideoWriter_fourcc(*'MJPG'), float(fps), (output_shape[1], output_shape[0]))

    count = 0
    image_files = []
    counter = 0
    index = 0
    while True:
        success, image = vidcap.read()
        if not success:
            print("Finished/Error in video: {}".format(input_video))
            break
        counter += 1
        if ((counter - 1) % 1) > 0:
            continue

        image = rotate_bound(image, rotation)

        if count % FPS == 0:
            with torch.no_grad():
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Apply transform to img
                img_trf = Image.fromarray(img)
                img_trf = transform(img_trf)
                img_trf = img_trf.unsqueeze(0)
                inputs = img_trf.to(device)
                # Apply model to get output
                outputs = model(inputs)

                # Prepare image input and output mask for blending
                i = inputs[0]
                i = i.cpu().numpy().transpose((1, 2, 0)) * 255
                i = i.astype(np.uint8)
                o = outputs[0]
                o = o.cpu().numpy().reshape(int(IMG_SIZE / 2), int(IMG_SIZE / 2)) * 255
                o = cv2.resize(o.astype(np.uint8), (output_shape[1], output_shape[0]))
                # Red color
                mask = np.zeros((output_shape[0], output_shape[1], 3)).astype(np.uint8)
                mask[:, :, 2] = o

                # Blend both mask and image
                org_resized_img = cv2.resize(image.astype(np.uint8), (output_shape[1], output_shape[0]))
                blend = cv2.addWeighted(mask, 0.3, org_resized_img, 0.7, 0)
                outvid.write(blend)

            index += 1
        count = count + 1

    outvid.release()
    print("Finished to processed video.")
