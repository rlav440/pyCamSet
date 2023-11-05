from __future__ import annotations
from functools import reduce
import numpy as np
from copy import copy
from matplotlib import pyplot as plt


class ImageDetection:
    """
    A data class that contains a numpy array of keys identifying points, and the image points
    at which that key was found.
    Enforces the correct implementation of an AbstractTarget inheretor's find_in_image function.
    """
    def __init__(self,  keys: np.ndarray|list|None=None, image_points: np.ndarray|list|None=None):
        """
        Takes a set of keys and detections,

        :param keys:
        :param image_points:
        """
        if not isinstance(keys, np.ndarray) and keys is not None:
            keys = np.array(keys)
        if not isinstance(image_points, np.ndarray) and image_points is not None:
            image_points = np.array(image_points)
        if keys is not None and image_points is not None:
            assert len(keys) == len(image_points), "Detected keys must be the same length as detected points"
            self.keys = keys
            self.image_points = image_points
            self.has_data = True
            self.data_len = len(keys)
        elif keys is None and image_points is None:
            self.has_data = False
        else:
            raise ValueError("A detection requires both identifying keys and detected image points.")


class TargetDetection:
    """
    This is a wrapper class that stores the information about which cameras saw which point in which image.
    At its core it is a numpy array that stores this data, while providing the ability to combine, iterate over and add
    data and detections to the object.

    Data is stored as
    | cam | im_num | key ... | data_x | data_y |
    Key is variable length.
    Cam is the index of the cam name in the input cam name list

    """

    def __init__(self, cam_names: list, data: np.ndarray | None = None, max_ims=0):
        # this class stores the information as a basic structure, while retaining the ability to be smoothly indexed.

        self.cam_names = cam_names
        if len(set(cam_names)) != len(cam_names):
            raise ValueError('input camera names must be unique')
        self._data = copy(data)
        self._get_methods = {
            'cam':self._get_cam, 'key':self._get_key, 'im_num':self._get_image_num, 'index':self._get_index,
        }

        self._update_buffer = []  # avoid copies when adding by writing to buffer
        self._max_ims = max_ims
        self._glomp_buffer()

    @property
    def max_ims(self):
        temp_data = int(np.max(self._data[:, 1])) + 1
        self._max_ims = max(temp_data, self._max_ims)
        return self._max_ims

    @max_ims.setter
    def max_ims(self, val):
        self._max_ims = val

    def get(self, **direction) -> TargetDetection:
        """
        Gets a subset of detections related to a certain camera, key or image number
        
        :param direction: A kwarg argument of either "cam", "key" or "im_num" and the associated data
        :return: A TargetDetection containing only the requested data
        """
        self._glomp_buffer()
        if len(direction) > 1:
            raise ValueError('Can only get one item at a time')
        key, target = next(iter(direction.items()))
        if key not in ['cam', 'key', 'im_num']:
            raise ValueError(f'{key} is not a gettable item: accepted are "cam", "key", or "im_num"')
        data = self._data[self._get_methods[key](target), :]
        if data.shape[0] == 0:
            data = None
        return TargetDetection(cam_names=self.cam_names, data=data, max_ims=self.max_ims)

    def delete_row(self, **direction):
        """
        :param direction: A cam, key, imnum or index, and the associated data to delete
        :return: A TargetDetection without hte deleted data.
        """
        self._glomp_buffer()
        if len(direction) > 1:
            raise ValueError('Can only get one item at a time')
        key, target = next(iter(direction.items()))
        if key not in ['cam', 'key', 'im_num', 'index']:
            raise ValueError(f'{key} is not a gettable item: accepted are "cam", "key", "im_num" or "index"')

        if not (isinstance(target, list) or isinstance(target, np.ndarray)):
            target = [target]

        masks = [self._get_methods[key](t) for t in target]
        final_mask = reduce(np.logical_or, masks)
        to_delete = np.where(final_mask)[0]
        new_data = np.delete(self._data, to_delete, axis=0)
        return TargetDetection(cam_names=self.cam_names, max_ims=self.max_ims, data=new_data)


    def get_cam_list(self) -> list[TargetDetection]:
        """
        :return: list of TargetDetections, each containing detections from a unique camera.
        """
        return [self.get(cam=name) for name in self.cam_names]

    def get_image_list(self) -> list[TargetDetection]:
        """
        :return: a list of TargetDetections, each containing detections from a unique image.
        """
        return [self.get(im_num=im_num) for im_num in range(int(self.max_ims))]

    def get_key_list(self) -> list[TargetDetection]:
        """
        :return: a list of target detections containing a unique index
        """
        unique_keys = np.unique(self.get_data(), axis=0)
        return [self.get(key=k) for k in unique_keys]

    def _get_cam(self, cam):
        """
        :param cam: A camera name
        :return: mask: a mask indicating the presence of the camera.
        """
        cam_index = self.cam_names.index(cam)
        mask = np.isclose(self._data[:, 0], cam_index)
        return mask

    def _get_key(self, key):
        """
        :param key: A key identifier
        :return: mask: a mask indicating the presence of the key
        """
        inds = range(2, 2 + len(key))
        key_blank = np.isclose(key, -1)
        masks = [np.isclose(self._data[:, idx], k) for idx, k in zip(inds, key) if not key_blank[idx-2]]
        mask = reduce(np.logical_and, masks)
        return mask

    def _get_image_num(self, im_num):
        """
        :param im_num: An image number
        :return: mask: a mask indicating data associated with the image number
        """
        mask = np.isclose(self._data[:, 1], im_num)
        return mask

    def _get_index(self, index_list):
        """
        :param index_list: A data index
        :return: mask: A mask only true at the requested index,
         so compatible with the other _get_* methods
        """
        mask = np.zeros(self._data.shape[0])
        mask[index_list] = True
        return mask

    def delete_col(self, col_id):
        """
        :param col_id: id of the column to delete
        :return: A target detection without the associated column.
        """
        new_data = np.delete(self._data, col_id, axis=1)
        return TargetDetection(cam_names=self.cam_names, max_ims=self.max_ims, data=new_data)

    def get_data(self) -> np.ndarray:
        """
        :return: Returns the internal data of the object, ensuring it is properly processed.
        """
        self._glomp_buffer()
        return self._data

    def __add__(self, other:TargetDetection) -> TargetDetection:
        """
        :param other: Another target detection with the same cameras
        :return: A single TargetDetection object containing the detections of both added TargetDetections
        """
        if not self.cam_names == other.cam_names:
            raise ValueError('To add detections, they must have consistent camera names. \n'
                             f'Detection 0 names: {self.names}\n'
                             f'Detection 1 names: {other.names}')
        self._glomp_buffer()
        other._glomp_buffer()
        if self._data.shape[0] == 0:
            if other._data.shape[0] == 0:
                shared_data = self._data
            else:
                shared_data = other._data
        else:
            if other._data.shape[0] != 0:
                shared_data = np.concatenate((self._data, other._data), axis=0)
            else:
                shared_data = self._data
        new_detection = TargetDetection(self.cam_names, shared_data)
        new_detection.max_ims = max(self.max_ims, other.max_ims)
        return new_detection

    def add_detection(self, cam_name, im_num, detection: ImageDetection) -> None:
        """
        Gets the elements of a detection and adds them to an input buffer.
        
        :param cam_name: The name of a detecting camera
        :param im_num: The image number of the detection.
        :param detection: The detection data, contained as an image detection.
        """
        ind = self.cam_names.index(cam_name)
        if detection.has_data:
            if detection.keys.ndim == 1:
                keys = detection.keys[..., None]
            else:
                keys = detection.keys
            observation = np.concatenate(
                [np.ones((detection.data_len, 2))*[ind, im_num], keys, detection.image_points]
                , axis=1)
            self._update_buffer.append(observation)

    def _glomp_buffer(self) -> None:
        """
        Incorporates the update buffer before use.
        """
        if self._update_buffer:
            if self._data is not None:
                self._data = np.append(self._data, np.concatenate(self._update_buffer, axis=0))
            else:
                self._data = np.concatenate(self._update_buffer, axis=0)
            self.max_ims = int(max(self.max_ims - 1, np.amax(self._data[:, 1])) + 1)
            self._update_buffer.clear()

    def sort(self, keys_to_sort: str|list[str], inplace=False) -> TargetDetection | None:
        """
        :param keys_to_sort: a single key, or multiple keys to sort by, with sorting order defined by list order.
        :param inplace: If true, replaces the internal data with the sorted array rather than returning new obj
        :return Either a sorted TargetDetection, or nothing if sorting in place
        """

        if not isinstance(keys_to_sort, list):
            keys_to_sort = [keys_to_sort]

        for item in keys_to_sort:
            if item not in ['cam', 'key', 'im_num']:
                raise ValueError(f"{item} is not an accepted sort key.\n"
                                 f"Accepted keys are: 'cam', 'key', or 'im_num'")

        data = self.get_data()
        lex_target = []
        for item in keys_to_sort[::-1]:
            if item == 'cam':
                temp = data[:, 0]

            elif item == 'im_num':
                temp = data[:, 1]
            elif item == "key":
                if self._data.shape[1] == 5: # 1D case
                    temp = data[:,2]
                else:
                    test = np.amax(data[:, 2:-2], axis=0) + 1
                    score_factor = np.append(np.cumprod(test[::-1])[::-1], 1)
                    temp = np.sum(score_factor[1:] * data[:, 2:-2], axis=1)
            else:
                raise ValueError("wrong key")
            lex_target.append(temp)
        inds = np.lexsort(lex_target)
        new_data = data[inds]
        if not inplace:
            return TargetDetection(self.cam_names, new_data, self.max_ims)
        else:
            self._data = new_data

    def features_per_im_per_cam(self) -> np.ndarray:
        """
        :return: The number of features detected per image per camera
        """
        n_cams = len(self.cam_names)
        n_ims = self.max_ims
        block = np.zeros((n_ims, n_cams))
        for cam_list in self.get_cam_list():
            cam_ind = int(cam_list.get_data()[0, 0])

            board_detected = 0
            im_lists = cam_list.get_image_list()
            for im_list in im_lists:
                datum = im_list.get_data()
                if datum is not None:
                    im_n = int(datum[0, 1])
                    total_seen = datum.shape[0]
                    block[im_n, cam_ind] = total_seen
        return block

    def return_flattened_keys(self, keydims) -> TargetDetection:
        """
        Flattens the key representations given the internal key dimensions.
        Unrolls from the last dimension, matching a numpy reshape.

        :param keydims: the maximum dimension of each key index.

        :return TargetDetection: A new target detection with a set of dim 1 keys.
        """
        if self.get_data().shape[1] == 5: #in this case, the data is already flat
            return self
        
        data = self.get_data().copy()
        padded_prod = np.append(keydims[1:], 1)
        prods = np.cumprod(padded_prod[::-1])[::-1]

        dim_1_keys = np.sum(data[:, 2:-2]*prods, axis=1).reshape((-1, 1))
        new_data = np.concatenate([data[:, :2], dim_1_keys, data[:, -2:]], axis=1)
        return TargetDetection(self.cam_names, new_data, self.max_ims)

    def parse_detections_to_reconstructable(self, draw_distribution=False):
        """
        Given the reference detection, detects which localised features can be triangulated, in which frame.
        It returns the subset of the data that can be used, and additional data that indicates the slices to use.
        It also calculates 
        
        :param draw_distribution: If true will draw an image number x feature number boolean plot, indicating which
            feature can be reconstructed in which image.

        :return feature_inds:
        :return im_dst:
        :return per_feature_count:
        :return reconstructable_data: 
        """
        data = self.sort(["keys", "images"]).get_data()
        # find keys that are reconstructable: that is keys that are seen by two+ cameras in a time point
        _, unique_key_inv, per_feature_count = np.unique(  # unique im num keys, etc
            data[:, 1:-2], axis=0, return_inverse=True, return_counts=True
        )

        viable_mask = per_feature_count > 1 #all features that are viabe
        data_recon_subset = data[viable_mask[unique_key_inv]] 
        
        #each task consists of all detections of a feature at a time point
        _, task_start_index, task_count = np.unique(
            data_recon_subset[:, 1:-2], axis=0, return_index=True, return_counts=True
        )
        sorted_task_count = task_count[np.argsort(task_start_index)]
        task_start_points = np.append(0, np.cumsum(sorted_task_count))


        # for a key/feature, how many images are in that key over all images in which that feature is visible
        _, feature_index = np.unique(data_recon_subset[:, 2:-2], axis=0)
        feature_inds = np.append(np.sort(feature_index), data_recon_subset.shape[0])
        im_dst = np.zeros((len(feature_inds) - 1, self.max_ims))
        idx = 0
        for i in range(len(feature_inds) - 1):
            j = 0
            while task_start_points[idx] < feature_inds[i + 1]:
                im_dst[i, j] = sorted_task_count[idx]
                idx += 1;
                j += 1

        per_feature_count = np.sum(im_dst > 0, axis=1)
        if draw_distribution:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(im_dst)
            ax[0].set_title('Feature visibility in cameras')
            ax[1].plot(per_feature_count, '.')
            ax[1].set_title('number visible images.')
            plt.show()

        return feature_inds, im_dst, per_feature_count, data_recon_subset
