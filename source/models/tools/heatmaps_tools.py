

#python
import os
import matplotlib.pyplot as plt

#numpy
import numpy as np

#opencv
import cv2


#local


def heatmaps_evaluation_manual(model, images, sv_path=None, sv_names=None):
    """
        Manually compute and returns the heatmap evaluation.
        To compare with the fcn method.
        See heatmaps evaluation for details.
    """

    img_shape = model.input_shape()



    return hms



def heatmaps_evaluation(fcn_model, images, sv_path=None, sv_names=None):
    """
        Evaluate the heatmaps for a given fcn model and set of images.
        If you provide a folder path and filenames to save the heatmaps,
        make sure the indexation of images and save_names matches.

        :param fcn_model: Fully convolutional model.
        :param images: Images to evaluate with the model.
        :param sv_path: Path to the folder in which to save the heatmaps.
        :param sv_names: Name under which to save the heatmaps.
        :return: Resulting heatmap for each input image.
        :type fcn_model: keras.model
        :type images: Numpy array of shape (nb_images, ) + image_shape
        :type sv_path: string
        :type sv_names: string list
        :rtype: numpy array, shapes depends of the fcn_model.
    """

    hms = fcn_model.predict(x=images, batch_size=4)

    if (sv_path != None) & (sv_names != None):
        if not os.path.exists(sv_path):
            os.mkdir(sv_path)
        for i in range(hms.shape[0]):
            for k in range(hms.shape[3]):
                plt.imsave('{}hm_cl{}_{}.png'.format(sv_path, k, sv_names[i]),
                    hms[i,:,:,k])

    return hms



def heatmaps_evaluation_video(fcn_model, cls, video_path, sv_path, sv_filename,
    fps_in=None, fps_out=None, verbose=0):
    """
        Compute the heatmaps of a fcn_model on a video.
        Returns the resulting heatmap video.

        :param fcn_model: Fully convolutional model.
        :param cls: Class to evaluate.
        :param video: Video to evaluate with the model.
        :param sv_path: Path to the folder in which to save the heatmaps.
        :param sv_filename: Name under which to save the heatmap video.
        :param fps_in: Maximum frame rate of images to process.
            If None, equals to the initial video fps. 
        :param fps_out: Fps setting of the ouptuted video.
            If None, is equal to fps_in.
        :param verbose: Vervose behavior.
        :return: Resulting heatmap for each input image.
        :type fcn_model: keras.model
        :type cls: int
        :type images: Numpy array of shape (nb_images, ) + image_shape
        :type sv_path: string
        :type sv_filename: string
        :type fps_in: int
        :type fps_out: int
        :type verbose: int
        :rtype: numpy array, shapes depends of the fcn_model.
    """

    if verbose:
        print('--Starting heatmaps_evaluation_video: {}.'.format(video_path))

    i_video = cv2.VideoCapture(video_path)
    width = int(i_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(i_video.get(cv2.CAP_PROP_FRAME_HEIGHT))


    if fps_in is None:
        fps_in = int(i_video.get(cv2.CAP_PROP_FPS))
    if fps_out is None:
        fps_out = fps_in

    o_path = os.path.join(sv_path, sv_filename)
    o_video = cv2.VideoWriter(o_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps_out, (width,height))

    if not i_video.isOpened():
        raise ValueError('Could not open video: {}.'.format(video_path))
    if not o_video.isOpened():
        raise ValueError('Could not open video: {}.'.format(o_path))


    ret = True
    k = 1
    pick = 0; pick_rate = max(fps_in // fps_out, 1)
    while ret:
        ret, frame = i_video.read()
        if ret:
            if verbose:
                print('Processing image --> {}'.format(k))
                k+=1

            if pick == 0:
                # big_frame = cv2.resize(frame, (width*2,height*2))
                imgs = np.zeros((1, ) + frame.shape, frame.dtype)
                imgs[0] = frame

                hms = heatmaps_evaluation(fcn_model, imgs)
                hm = hms[0,:,:,cls]
                hm -= np.min(hm)
                hm /= np.max(hm)
                print(hm.shape)
                # plt.imsave('{}video_testing_{}.png'.format(sv_path, k), resized)

                resized = cv2.resize(hm, (width, height))
                resized = resized.reshape(resized.shape[0], resized.shape[1], 1)
                frame = (frame * resized).astype('uint8')

                o_video.write(frame)

            pick = (pick + 1) % pick_rate
                

    return o_video