===========================
AbstractTarget
===========================


Introduction
============

The AbstractTarget is an abstract base class that all calibration targets extend from.
All specific implementations define both the location of the feature points of the target, and a implement the "find_in_image" method.
Other methods left to implement are defining a printable net, and optionally visualisation of the target.


Examples
========

Implementing ChArUco 
--------------------

The ChArUco target is a target defined by the openCV library.
It defines methods for detecting and calibrating a camera, and is robust and reliable.
It is also well suited for calibration of multiple cameras, as partial views of the target still result in valid detections.
However, within the openCV framework it can't be used to jointly calibrate 3+ cameras, so we can wrap the base target with the AbstractTarget to enable this.
This starts with the input:

::

   class ChArUco(AbstractTarget):


There are 2 tasks that need to be completed to implement the wrapper.
#. An initial set up that creates the point data array representing the locations of the points.
#. Implementing the find_in_image function, ensuring that the code returns an ImageDetection object.


The first task takes the form of the __init__ method of the class.

::
    
    def __init__(self, num_squares_x, num_squares_y, square_size, adict=cv2.aruco.DICT_4X4_1000):
        super().__init__(inputs=locals())

        # define checker and marker size
        square_size = square_size
        marker_size = 0.8 * square_size  # 80% of the square size
        # convert to meters
        squares_length = square_size / 1000
        marker_length = marker_size / 1000

        # Create the dictionary for the Charuco board
        self.a_dict = cv2.aruco.Dictionary_get(adict)
        # Create the Charuco board
        self.board = cv2.aruco.CharucoBoard_create(num_squares_x, num_squares_y, squares_length, marker_length,
                                                   self.a_dict)
        self.point_data = self.board.chessboardCorners.squeeze().astype(np.float64)

        self._process_data()

There are three sections of this code. 
The first is the call to the super initialisation. 
This call passes the input arguments to the super class, where they are stored as a reference for regeneration of the target.
The middle section creates a ChArUco board, adding it as a class member (where it can later be used for detection), and defines the point_data array.
The final call to _process_data() generates a local coordinate system for each face of the target (although there is one face for this target), which is used for initial calibrations.

With this defined, we can write the find_in_image class method.
::
   
    def find_in_image(self, image, draw=False, camera: Camera = None, wait_len=1) -> ImageDetection:
        params = aruco.DetectorParameters_create()
        params.minMarkerPerimeterRate = 0.01
        corners, ids, _ = aruco.detectMarkers(image, self.a_dict, parameters=params)

        if len(corners) == 0:
            return ImageDetection() # return an empty detection

        if draw:
            display_im = image.copy()
            target_size = [640, 480]
            d_f = int(min(np.array(display_im.shape[:2]) / target_size))
            display_im = downsample_valid(display_im, d_f).astype(np.uint8)
            if display_im.ndim == 2:
                display_im = np.tile(display_im[..., None], (1, 1, 3))

            # aruco.drawDetectedMarkers(im_idea, np.array(corners)/d_f, ids)
        use_cam = camera is not None
        n, c_corners, c_ids = aruco.interpolateCornersCharuco(
            corners,
            ids,
            image,
            self.board,
            camera.intrinsic if use_cam else None,
            camera.distortion_coefs if use_cam else None,
        )
        if n == 0:
            return ImageDetection()

        if draw:
            aruco.drawDetectedCornersCharuco(
                display_im,
                np.array(c_corners) / d_f,
                c_ids,
            )

            cv2.imshow('detections', display_im)
            cv2.waitKey(wait_len)

        return ImageDetection(c_ids[:, 0], c_corners[:, 0])

This is a standard implementation of target detection using the ChArUco board, with an optional draw flag.
With these methods defined, the new ChArUco target can be used as an input to the calibration library, allowing the use of ChArUco targets for calibrating n-camera systems.
