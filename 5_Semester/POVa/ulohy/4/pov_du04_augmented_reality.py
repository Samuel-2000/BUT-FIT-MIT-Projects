# https://longervision.github.io/2017/03/12/ComputerVision/OpenCV/opencv-external-posture-estimation-ArUco-board/

import numpy as np
import cv2
import argparse


def parseargs():
    parser = argparse.ArgumentParser('Train recurrent network LM.')
    parser.add_argument('-v', '--video-file', help='Process a video file. If not specified, opens first camera.')
    parser.add_argument('-c', '--camera-params', required=True, help='File from calibrate.py.')
    parser.add_argument('--chessboard-rows', default=6, type=int, help='Number of chessboard inner corners.')
    parser.add_argument('--chessboard-cols', default=4, type=int, help='Number of chessboard inner corners.')
    args = parser.parse_args()
    return args


# Wave simulation. This is not part of the assignment and there are now errors here.
# from https://beltoforion.de/en/recreational_mathematics/2d-wave-equation.php
class WaveSim:
    def __init__(self, width=64, height=64):
        self.u = np.zeros((3, height, width))         # The three dimensional simulation grid
        self.c = 0.5                                # The "original" wave propagation speed
        self.h = 1  # spatial step width
        self.k = 1  # time step width
        self.alpha = np.zeros((height, width))        # wave propagation velocities of the entire simulation domain
        self.alpha[...] = ((self.c*self.k) / self.h)**2  # will be set to a constant value of tau

        y = np.random.randint(5, height - 5)
        x = np.random.randint(5, width - 5)
        self.u[0, x - 2:x + 2, y - 2:y + 2] = 100

    def update(self):
        self.u[2] = self.u[1]
        self.u[1] = self.u[0]
        s1 = self.u.shape[1]
        s2 = self.u.shape[2]

        self.u[0, 1:s1-1, 1:s2-1] = self.alpha[1:s1-1, 1:s2-1] * (self.u[1, 0:s1-2, 1:s2-1] +
                                        self.u[1, 2:s1,   1:s2-1] +
                                        self.u[1, 1:s1-1, 0:s2-2] +
                                        self.u[1, 1:s1-1, 2:s2] - 4*self.u[1, 1:s1-1, 1:s2-1]) \
                                        + 2 * self.u[1, 1:s1-1, 1:s2-1] - self.u[2, 1:s1-1, 1:s2-1]

        # Not part of the wave equation but I need to remove energy from the system.
        # The boundary conditions are closed. Energy cannot leave and the simulation keeps adding energy.
        self.u[0, 1:s1-1, 1:s2-1] *= 0.99

        if np.random.uniform() < 0.02:
            y = np.random.randint(5, s1-5)
            x = np.random.randint(5, s2-5)
            self.u[0, x-2:x+2, y-2:y+2] = 120

    def get_points(self, width=5, height=5):
        x = np.linspace(0, width, num=self.u.shape[2] - 2)
        y = np.linspace(0, height, num=self.u.shape[1] - 2)
        xv, yv = np.meshgrid(x, y, indexing='ij')
        coords = np.stack([xv, yv, self.u[0, 1:-1, 1:-1]/400 - 1], axis=2)
        return coords.reshape(-1, 3)


# Just draw points 2D, nothing special here and no errors
def draw_points(img, points, colors, color_low=np.asarray([255, 0, 0]), color_high=np.asarray([0, 0, 255]),  point_size=2):
    points = points.astype(int)
    for i, p in enumerate(points[:, 0]):
        color = colors[i] * color_low + (1-colors[i]) * color_high
        color = tuple(color.astype(np.uint8).tolist())
        cv2.circle(img, (p[0], p[1]), radius=point_size, color=color, thickness=-1)


def main():
    args = parseargs()

    # the wave simulation
    wave_sim = WaveSim(64, 64)

    # The size of chessboard
    chessboard_rows = args.chessboard_rows
    chessboard_cols = args.chessboard_cols

    npzfile = np.load(args.camera_params)
    mtx = npzfile['mtx']
    dist = npzfile['dist']

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001)

    board_points = np.zeros((args.chessboard_cols * args.chessboard_rows, 3), np.float32)
    board_points[:, :2] = np.mgrid[0:args.chessboard_rows, 0:args.chessboard_cols].T.reshape(-1, 2)

    if args.video_file:
        capture = cv2.VideoCapture(args.video_file)
    else:
        capture = cv2.VideoCapture()

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        wave_sim.update()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.blur(frame_gray, (5, 5))
        ret, corners = cv2.findChessboardCorners(frame_gray, (args.chessboard_rows, args.chessboard_cols), None)
        if ret:
            corners = cv2.cornerSubPix(frame_gray, corners, (7, 7), (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (args.chessboard_rows, args.chessboard_cols), corners, ret)

            # Find the rotation and translation vectors.
            #ret, rvecs, tvecs = cv2.solvePnP(corners, board_points, mtx, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            ret, rvecs, tvecs = cv2.solvePnP(board_points, corners, mtx, None, flags=cv2.SOLVEPNP_ITERATIVE)

            # project 3D points to image plane
            points = wave_sim.get_points()
            #points_img, jac = cv2.projectPoints(points, tvecs, rvecs, mtx, dist)
            points_img, jac = cv2.projectPoints(points, rvecs, tvecs, mtx, dist)

            # get point colors from z coordinate
            colors = np.clip((points[:, 2] + 1) * 2 + 0.5, 0, 1)

            draw_points(frame, points_img, colors)

        cv2.imshow('img', frame)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('x'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
