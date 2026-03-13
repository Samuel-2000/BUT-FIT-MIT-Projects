import argparse
import glob
import cv2
import os
import logging

import numpy as np


def parseargs():
    parser = argparse.ArgumentParser('Calibrate camera from chessboard images.')
    parser.add_argument('-p', '--path', required=True, help='Path to images. All .jpg files from this directory will be used for calibration.')
    parser.add_argument('-o', '--output-file', required=True, help='Where to save the calibration camera parameters.')
    parser.add_argument('--chessboard-rows', default=6, type=int, help='Number of chessboard inner corners.')
    parser.add_argument('--chessboard-cols', default=4, type=int, help='Number of chessboard inner corners.')
    args = parser.parse_args()
    return args


def main():
    args = parseargs()

    # The size of the chessboard. Only the inner corners (not the chessboard edge).
    chessboard_rows = args.chessboard_rows
    chessboard_cols = args.chessboard_cols

    # Termination criteria for subpixel chessboard corner localization.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    image_points = []

    # Iterate over all .jpg images in args.path directory
    for image_path in glob.glob(os.path.join(args.path, '*.jpg')):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.error(f'Unable to read image"{image_path}".')
            exit(-1)

        ret, corners = cv2.findChessboardCorners(img, (chessboard_rows, chessboard_cols), None)
        if not ret:
            logging.warning(f'Failed to find chessboard in "{image_path}".')
            continue

        # Refine the corner position to subpixel precision.
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        image_points.append(corners)

        img = np.stack([img, img, img], axis=2)
        cv2.drawChessboardCorners(img, (chessboard_rows, chessboard_cols), corners, ret)
        cv2.imshow('output', img)
        cv2.waitKey(50)

    # The chessboard is the center of the universe :) - it is the stable object (reference coordinate frame)
    # and the camera is moving around it.
    # Prepare 3D coordinates of the chessboard corners - the z coordinate of chessboard is 0
    # the coordinates are (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    board_points = np.zeros((chessboard_cols * chessboard_rows, 3), np.float32)
    board_points[:, :2] = np.mgrid[0:chessboard_rows, 0:chessboard_cols].T.reshape(-1, 2)

    # These are board 3D coordinates for each view
    # world_boards_points = [board_points for i in range(4)]
    world_boards_points = [board_points for _ in range(len(image_points))]

    # Get the camera calibration
    # mtx - intrinsic camera params as 3x3 matrix - contains focal lengths and projection centers
    # dist - camera distortion coefficients
    # rvecs, tvecs - pose for each chesboard view
    img_shape = (img.shape[1], img.shape[0])
    #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(image_points, world_boards_points, img_shape, None, None)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_boards_points, image_points, img_shape, None, None)

    # Save camera calibration to file, we will need this later.
    np.savez(args.output_file, mtx=mtx, dist=dist)

    # The question is: If you have some distortion parameters, what would be good crop region after the distortion is removed.
    # The fourth parameter controlls how much the image should be cropped (values 0-1)
    mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (img.shape[1], img.shape[0]), 0.2, img_shape)

    # Undistort the image (try to recover perfect projective view with no camera distortions.
    # mapx, mapy contain mapping for each pixel and cv2.remap performs fast geometric transformation using these per-pixel displacements
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (img.shape[1], img.shape[0]), cv2.CV_32FC2)
    img_undistorted = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # Crop and show the undistorted image.
    x, y, w, h = roi
    img_undistorted = img_undistorted[y:y+h, x:x+w]
    cv2.imshow('undistorted', img_undistorted)
    cv2.imwrite('calibresult.png', img_undistorted)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()