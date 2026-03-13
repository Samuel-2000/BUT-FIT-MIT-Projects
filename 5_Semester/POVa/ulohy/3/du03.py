# coding: utf-8

import numpy as np
import cv2

# Grabcut is an image segmentation algorithm which is best used in interactive segmentation.
# 1) It starts from some seed regions which represent foreground and background.
# 2) It creates a models representing appearance of FG/BG pixels - usually a gaussian mixture model of pixel color
# 3) Computes background/foreground probabilities for each pixel in the image.
# 4) Recomputes best background/foreground partition with added smoothness constraints using minimum graph cut algorithm.
#
# Some implementations iterate several times through steps 2-4, (hopefully) improving the color models and the segmentation.
# GrabCut version in OpenCV can start with a mask indicating which pixels definitely belong to BG/FG
# and which probably belong to BG/FG. This is used in this assignment.
# Alternatively it can start with a rectangle, where it is expected that the center of the rectangle contains the object
# and outside of the rectangle is only background.
#
# Original paper: Carsten Rother, Vladimir Kolmogorov, and Andrew Blake. 2004. "GrabCut": interactive foreground
# extraction using iterated graph cuts. ACM Trans. Graph. 23, 3 (August 2004), 309-314. DOI: https://doi.org/10.1145/1015706.1015720
#
# Video lecture:
# * 2019 - Graph cuts from 1:26 - https://video1.fit.vutbr.cz/av/records.php?id=51372&categ_id=1598
#
# Good reading options are:
# * https://pyimagesearch.com/2020/07/27/opencv-grabcut-foreground-segmentation-and-extraction/
# * https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html,
# * https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga909c1dda50efcbeaa3ce126be862b37f
	

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        epilog='Grab cut demonstration. ' +
        'Manually crop rectangle by mouse drag. ' +
        'An interactive Grab cut segmentation session is run on the selected crop. ' +
        'Label foreground and background pixels as needed with mouse. ' +
        'Use "f" and "b" keys to switch to foreground (f) respective background (b) annotation. ' +
        'Use "space" to update the segmentation. Exit by pressing Escape key.')
    parser.add_argument('-i', '--image', required=True,
                        help='Image file name.')
    args = parser.parse_args()
    return args


class RectangleCropCallback(object):
    """
    This callback handles mouse interaction in the original image window. It allows a user to crop the image.
    """
    def __init__(self):
        self.first_point = None
        self.second_point = None
        self.cropping_now = False
        self.finished_cropping = False

    def mouse_callback(self, event, x, y, flags, param):
        # If the left mouse button is clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed.
        if event == cv2.EVENT_LBUTTONDOWN:
            self.first_point = self.second_point =(int(x), int(y))
            self.cropping_now = True

        # If cropping, update rectangle.
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cropping_now:
                self.second_point = (int(x), int(y))

        # Finish cropping when left mouse button is released.
        elif event == cv2.EVENT_LBUTTONUP:
            if self.cropping_now:
                self.second_point = (int(x), int(y))
                self.cropping_now = False
                self.finished_cropping = True


class GrabCutCallback(object):
    """
       This callback handles mouse interaction in the segmentation image window.
       It allows a user to manually select background and foreground pixels.
       """
    def __init__(self, image):
        self.img = image
        # Create inital foreground/background mask.
        # Lets say that all pixels probably contain background.
        # Grab cut starts from this mask.
        self.mask = np.full(
            shape=image.shape[:2], fill_value=cv2.GC_PR_BGD, dtype=np.uint8)
        self.annotating_foreground = True
        self.drawing_active = False

    def render_mask_to_image(self):
        img = self.img.copy()
        # Mark foreground in the image.
        img[:, :, 1][self.mask == cv2.GC_FGD] = 0
        img[:, :, 1][self.mask == cv2.GC_PR_FGD] = img[:, :, 1][self.mask == cv2.GC_PR_FGD] / 2
        # Mark background in the image.
        img[:, :, 2][self.mask == cv2.GC_BGD] = 0
        img[:, :, 2][self.mask == cv2.GC_PR_BGD] = img[:, :, 2][self.mask == cv2.GC_PR_BGD] / 2
        return img

    def mouse_callback(self, event, x, y, flags, param):
        # Start drawing into mask on mouse button press.
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing_active = True
            if self.annotating_foreground:
                self.mask[int(y), int(x)] = cv2.GC_FGD
            else:
                self.mask[int(y), int(x)] = cv2.GC_BGD

        # Draw mask pixels on mouse move.
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing_active:
                if self.annotating_foreground:
                    self.mask[int(y), int(x)] = cv2.GC_FGD
                else:
                    self.mask[int(y), int(x)] = cv2.GC_BGD

        # Stop drawing annotation on button release.
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing_active = False


def main():
    args = parse_arguments()

    # Will be using two windows.
    # One for cropping. One for segmentation.
    cv2.namedWindow('image')
    cv2.namedWindow('segmentation', cv2.WINDOW_NORMAL)

    input_image = cv2.imread(args.image)

    # Init mouse callback for the cropping image window.
    crop_cb = RectangleCropCallback()
    # Assign the callback crop_cb to the 'image' window as a mouse callback.
    # In python crop_cb.mouse_callback is a bound function
    # - it remembers the crop_cb object as self. A bound function can be passed to any function as other python object.
    cv2.setMouseCallback('image', crop_cb.mouse_callback)

    # this will hold a segmentation mouse callback when the image is cropped
    segment_cb = None

    while(True):
        # Create image copy as we will draw into it and we don't want to change the original image.
        tmp_img = input_image.copy()

        # If cropping in progress, draw the region.
        if crop_cb.cropping_now:
            # Draw rectangle between crop_cb.first_point and crop_cb.second_point.
            # Use color (255, 0, 0). You can use cv2.rectangle().
            # Draw into tmp_img.
            cv2.rectangle(tmp_img, crop_cb.first_point, crop_cb.second_point, (255, 0, 0), 2)

        # Start segmentation when user finishes cropping.
        if crop_cb.finished_cropping:
            crop_cb.finished_cropping = False

            # Get rectangular crop region.
            x1 = min(crop_cb.first_point[0], crop_cb.second_point[0])
            y1 = min(crop_cb.first_point[1], crop_cb.second_point[1])
            width = abs(crop_cb.second_point[0] - crop_cb.first_point[0])
            height = abs(crop_cb.second_point[1] - crop_cb.first_point[1])
            crop = input_image[y1:y1 + height, x1:x1 + width, :]  # this makese reference to the original image (it is not a copy).

            segment_cb = GrabCutCallback(crop)
            # Assign the callback segment_cb to the 'segmentation' window.
            cv2.setMouseCallback('segmentation', segment_cb.mouse_callback)

        # Display current segmentation if the segmentation is "live".
        if segment_cb:
            cv2.imshow('segmentation', segment_cb.render_mask_to_image())

        # Display the original image (with cropping rectangle).
        cv2.imshow('image', tmp_img)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:   # 27 this is the Escape key
            break
        elif key == ord('f') and segment_cb:  # switch to forground selection
            segment_cb.annotating_foreground = True
        elif key == ord('b') and segment_cb:  # switch to background selection
            segment_cb.annotating_foreground = False
        elif key == ord(' ') and segment_cb:  # recompute segmentation
            # These will hold Gaussian Mixture Models for the background and foreground color distribution.
            # These models are not used in this code, but can reused in repeated grabCut calls.
            bg_color_model = None
            fg_color_model = None
            # Run cv2.grabCut() on segment_cb.img and segment_cb.mask.
            # Init the algorithm with the current content of segment_cb.mask. Run it for 2 iterations.
            cv2.grabCut(segment_cb.img, segment_cb.mask, None, bg_color_model, fg_color_model, 2, cv2.GC_INIT_WITH_MASK)

    # If a segmentation was computed, process it.
    if segment_cb:
        # Create a binary mask from the segmentation in segment_cb.mask which was produced by grabCut.
        # True/1 should be assigned to pixels with values cv2.GC_FGD and cv2.GC_PR_FGD.
        # Pixels with values cv2.cv2.GC_PR_BGD and cv2.GC_BGD should be assigned False/0.
        binary_mask = np.where((segment_cb.mask == cv2.GC_FGD) | (segment_cb.mask == cv2.GC_PR_FGD), 1, 0)


        # Add some random foreground noise pixels to binary_mask.
        positions0 = np.random.random_integers(binary_mask.shape[0] - 1, size=100)
        positions1 = np.random.random_integers(
            binary_mask.shape[1] - 1, size=positions0.size)
        binary_mask[positions0, positions1] = 1

        # Add some random background noise pixels to binary_mask.
        positions0 = np.random.random_integers(binary_mask.shape[0] - 1, size=100)
        positions1 = np.random.random_integers(
            binary_mask.shape[1] - 1, size=positions0.size)
        binary_mask[positions0, positions1] = 0

        # Show the noisy mask.
        binary_mask = np.uint8(binary_mask)
        cv2.imshow('noisy mask', binary_mask * 255)

        # Remove lonely (noisy) foreground pixels from binary_mask.
        # Use morfological operation open (erosion followed by dilatation).
        # Use 'kernel' in the operation.
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Remove small holes from binary_mask (noisy background pixels).
        # Use morfological operation close - dilatation followed by erosion.
        # Use 'kernel' in the operation.
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        cv2.imshow('repaired mask', np.uint8(binary_mask) * 255)

        # Mask foreground pixels of the original image.
        # Set background background pixels of the image to back color.
        # Use 'binary_mask' as the mask and segment_cb.img as the source image.
        masked_foreground_image = segment_cb.img * binary_mask[:, :, np.newaxis]

        cv2.imshow('masked foreground', masked_foreground_image)

        # Compute distance transform in order to
        # highlight all pixels 20px distant from the foreground (1 pixels in binary_mask)
        # Opencv has a function which computes distance transform efficiently.
        distances = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

        outline_20_px = np.uint32(distances) == 20
        cv2.imshow('distance 20', np.uint8(outline_20_px) * 255)
        cv2.waitKey()

        # Compute vertical and horizontal projection of the foreground in binary_mask.
        # The projections can be computed by summing mask pixel in horizontal lines respective vertical columns.
        # Use matplotlib.pyplot to plot the projection graphs.
        # Use subplot() to put both graphs into a single window.
        horizontal_projection = np.sum(binary_mask, axis=1)
        vertical_projection = np.sum(binary_mask, axis=0)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(horizontal_projection, range(len(horizontal_projection)))
        plt.gca().invert_yaxis()
        plt.title('Horizontal Projection')
        plt.xlabel('Sum of foreground pixels')
        plt.ylabel('Row index')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(len(vertical_projection)), vertical_projection)
        plt.title('Vertical Projection')
        plt.xlabel('Column index')
        plt.ylabel('Sum of foreground pixels')
        
        plt.tight_layout()
        plt.show()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
