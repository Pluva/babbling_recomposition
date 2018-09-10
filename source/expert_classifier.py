"""
    Expert classifier functions.
    Usefull to generate artificial datasets.


"""

#python
import matplotlib
import os

#numpy
import numpy as np

#opencv
import cv2

#local


def compare_to_background(image1, image2, treshold=None, grayscale=False, verbose=0):
    """
        WARNING: IMPLEMENTATION HAS TO BE IMPROVED FOR BETTER PERFORMANCE.
        Compare two given images, returns a mask of difference.
        Zeros for identical pixels, one for different.
        Images have to be channel last.

        :param image1: Input image 1.
        :param image2: Input image 2.
        :param treshold: Comparison treshold.
        :param grayscale: Grayscale images.
        :param verbose: Verbose behavior [default=0].
        :return: A 2D mask.
        :type image1: cv mat
        :type image2: cv mat.
        :type treshold:
        :type grayscale: boolean.
        :type verbose: int
        :rtype: cv mat
    """

    if grayscale:
        ret = cv2.absdiff(image1, image2)
        # ret = cv2.adaptiveThreshold(ret, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,5,4)
        _, ret = cv2.threshold(ret, 50, 255, cv2.THRESH_BINARY)
    else:
        #TO CHANGE
        img1 = matplotlib.colors.rgb_to_hsv(image1)[:,:,0]
        img2 = matplotlib.colors.rgb_to_hsv(image2)[:,:,0]
        ret = (cv2.absdiff(img1, img2) > treshold).astype(int)

    return ret

def get_boudingrect_foreground(diff, min_size, group=True, make_square=False, verbose=0):
    """
    """

    _, contours, _ = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for contour in contours:
        (x,y,h,w) = cv2.boundingRect(contour)
        if (w > min_size) & (h > min_size) & (w < 4*min_size) & (h < 4*min_size):
            if make_square:
                #carefull won't stand the case where you hit the walls
                if w > h:
                    x = max(x - (w-h//2), 0)
                    h += min(w - h, diff.shape[0] - x - 1)
                elif h > w:
                    y = max(y - (h-w//2), 0)
                    w += min(h - w, diff.shape[1] - y - 1)



            rects.append((x,y,w,h))



    # if group:
    #     rects = np.array(rects)
    #     print(rects)
    #     rects = cv2.groupRectangles(rects, 1, eps=0)

    return rects

# def get_boudingsquare_foreground(diff, min_size, group=True, verbose=0):
#     """
#     """
#     _, contours, _ = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     squares = []
#     for contour in contours:
#         center, radius = cv2.minEnclosingCircle(contours)
#         if radius > min_size:
#             x, y = center
#             squares.append([x-radius,y-radius,radius*2, radius*2])

#     return squares


def get_images_boundingrect(image, rects):
    """
    """
    return


def predict_on_video_stream(video_path, background, model, sv_path, sv_filename, fps_in=None, fps_out=None, verbose=0):
    """
    """

    if verbose:
        print('--Starting predict_on_video_stream: {}.'.format(video_path))

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
    pick = 0; pick_rate = max(fps_in//fps_out, 1)
    while ret:
        ret, frame = i_video.read()
        if ret:
            if verbose:
                print('Processing image --> {}'.format(k))
                k+=1

            if pick == 0:
                frame_gray = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
                frame_gray = frame_gray.astype('uint8')
                mask = compare_to_background(image1=frame_gray, image2=background, grayscale=True)
                rects = get_boudingrect_foreground(mask, min_size=100, make_square=False)

                for (x,y,w,h) in rects:
                    sub_img = frame[y:y+w, x:x+h]
                    sub_img = cv2.resize(sub_img, (128,128))

                    sub_imgs = np.zeros((1, ) + sub_img.shape, dtype=sub_img.dtype)
                    sub_imgs[0] = sub_img
                    pred = model.predict(sub_imgs)
                    if pred[0][0] >= 0.95:
                        frame = cv2.rectangle(frame,(x,y),(x+h,y+w),(0,0,300),3)
                    elif pred[0][0] < 0.3:
                        frame = cv2.rectangle(frame,(x,y),(x+h,y+w),(300,0,0),3)
                    else:
                        frame = cv2.rectangle(frame,(x,y),(x+h,y+w),(0,300,0),3)


                o_video.write(frame)

            pick = (pick + 1) % pick_rate

    return o_video