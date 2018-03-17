import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math
from moviepy.editor import VideoFileClip


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def lane_finding(image, low_threshold=50, high_threshold=150,
                 rho=1, theta=np.pi / 180, threshold=10, min_line_len=10, max_line_gap=25):
    height, width, c = image.shape

    # TO GRAYSCALE
    work = grayscale(image)

    # GAUSSIAN BLUR
    kernel_size = 5
    work = gaussian_blur(work, kernel_size)

    # EDGE DETECTION
    work = canny(work, low_threshold, high_threshold)
    edges = work.copy()

    # REGION OF INTEREST
    vertices = np.array([[(70, height), (450, 320), (510, 320), (width - 50, height)]], dtype=np.int32)
    work = region_of_interest(work, vertices)
    masked = work.copy()

    # draw mask onto the original for tunning
    #     im_mask_borders = image.copy()
    #     num_vert = vertices.shape[1]
    #     for i in range(1, num_vert + 1):
    #         cv2.line(im_mask_borders, tuple(vertices[0, i-1, :]), tuple(vertices[0, i % num_vert, :]), [0, 0, 255], 5)
    #     plt.imshow(im_mask_borders)
    #     plt.show()

    # HOUGH TRANSFORM (PROBABILISTIC): FIND LINES
    im_lines = hough_lines(work, rho, theta, threshold, min_line_len, max_line_gap)
    lines = cv2.HoughLinesP(work, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    # filter out lines with infinite or extreme derivatives
    der = (lines[:, :, 2] - lines[:, :, 0]) / (lines[:, :, 3] - lines[:, :, 1])
    der_idx = np.logical_and(np.logical_not(np.isinf(der)), np.logical_and(np.abs(der) > 1, np.abs(der) < 3))
    lines = lines[der_idx, ...]
    der = der[der_idx, ...]
    # plt.hist(der, 50)
    # plt.show()

    # divide lines into left and right, average and extrapolate
    left_lines_idx = (der < 0).squeeze()
    right_lines_idx = np.logical_not(left_lines_idx)
    left_lines = lines[left_lines_idx, ...].squeeze()
    right_lines = lines[right_lines_idx, ...].squeeze()

    # fit a line through detected points
    left_x = np.hstack((left_lines[..., 0], left_lines[..., 2]))
    left_y = np.hstack((left_lines[..., 1], left_lines[..., 3]))
    left_p = np.polyfit(left_x, left_y, 1)
    left_x = [70, 450]
    left_y = np.polyval(left_p, left_x)

    right_x = np.hstack((right_lines[..., 0], right_lines[..., 2]))
    right_y = np.hstack((right_lines[..., 1], right_lines[..., 3]))
    right_p = np.polyfit(right_x, right_y, 1)
    right_x = [width - 50, 510]
    right_y = np.polyval(right_p, right_x)

    lines = np.array([[[left_x[0], left_y[0], left_x[1], left_y[1]]],
                      [[right_x[0], right_y[0], right_x[1], right_y[1]]]], dtype=np.int32)

    # DRAW LINES
    line_img = np.zeros((work.shape[0], work.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, thickness=10)

    return weighted_img(line_img, image, 1.), (edges, im_lines)


def process_test_images():
    # RUN LANE FINDING on all test images
    in_dirname = 'test_images/'
    out_dirname = 'test_images_output/'

    if not os.path.isdir(out_dirname):
        # create output directory if it doesn't exist
        os.makedirs(out_dirname)
    listdir = os.listdir(in_dirname)

    for im_filename in listdir:
        in_path = in_dirname + im_filename
        image = mpimg.imread(in_path)
        im_lanes, rest = lane_finding(image,
                                      low_threshold=100, high_threshold=250,
                                      threshold=25, min_line_len=40, max_line_gap=5)

        out_path = out_dirname + im_filename
        print('Writing {} ...'.format(out_path))
        mpimg.imsave(out_path, im_lanes, format='jpg')
        print('Done!')


def process_image(image):
    result, rest = lane_finding(image,
                                low_threshold=120, high_threshold=240,
                                rho=1, theta=1*np.pi/180, threshold=15, min_line_len=50, max_line_gap=60)
    return result


def process_test_video(filename):
    input_file = 'test_videos/' + filename
    output_file = 'test_videos_output/' + filename

    in_clip = VideoFileClip(input_file).subclip(6, 12)
    # pass in the handle of the image processing function applied on each frame
    out_clip = in_clip.fl_image(process_image)
    # save processed video clip
    out_clip.write_videofile(output_file, audio=False)


def analyze_image(filename):
    im = mpimg.imread('test_images/' + filename)
    im_lanes, rest = lane_finding(im,
                                  low_threshold=120, high_threshold=240,
                                  rho=2, threshold=40, min_line_len=150, max_line_gap=60)

    plt.figure()
    plt.imshow(im_lanes)

    plt.figure()
    plt.imshow(rest[0], cmap='gray')

    plt.figure()
    plt.imshow(rest[1])
    plt.show()


if __name__ == '__main__':
    # process_test_images()

    # analyze_image('problematic4.jpg')
    process_test_video('solidYellowLeft.mp4')
    process_test_video('solidWhiteRight.mp4')


