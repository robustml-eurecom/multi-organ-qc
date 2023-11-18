import os
import numpy as np
from skimage import io
import plotly.graph_objects as go


def get_dict_with_key(key, list_of_dicts):
    for dictionary in list_of_dicts:
        if key in dictionary:
            return dictionary[key]
    return None


def find_non_matching_pairs(mask_paths, image_paths):
    """Takes a list of paths for masks and images, and returns the paths of each which do not have a matching pair in the other.
    Eg. If an image doesn't have a corresponding mask, then this image path will be returned. 

    Parameters
    ----------
    mask_paths: list[os.PathLike]
        The list of masks paths.
    image_paths: lsit[os.PathLike]
        The list of image paths.

    Returns
    -------
    (list, list) : missing_masks_pairs, missing_image_pairs
    """
    # Retrieving directory names
    mask_set = set([os.path.dirname(path) for path in mask_paths])
    image_set = set([os.path.dirname(path) for path in image_paths])
    # Finding unmatched ones
    missing_files = mask_set.symmetric_difference(image_set)
    # Retrieving the path of unmatched
    missing_mask_pairs = [p for p in mask_paths if os.path.dirname(p) in missing_files and "mask.nii.gz" in p]
    missing_image_pairs = [p for p in image_paths if os.path.dirname(p) in missing_files and "image.nii.gz" in p]

    return missing_mask_pairs, missing_image_pairs

def ignore_directories(mask_paths, image_paths):
    """
    Takes as input two sets of paths and removes the ones who don't have a matching one according to find_non_matching_pairs.
    
    Parameters
    ----------
    mask_paths: list[os.PathLike]
        The list of masks paths.
    image_paths: lsit[os.PathLike]
        The list of image paths.

    Returns
    -------
    np.ndarray : the list containing the directories to ignore
    """
    missing_mask_pairs, missing_image_pairs = find_non_matching_pairs(mask_paths, image_paths)
    missing_image_pairs = [os.path.dirname(p) for p in missing_image_pairs]
    missing_mask_pairs = [os.path.dirname(p) for p in missing_mask_pairs]

    ignore_directories = np.unique(missing_mask_pairs, missing_image_pairs)

    return ignore_directories


class Visualizer():
    
    def __init__(self, rand_img):
        self.rand_img = rand_img
        self.r, self.c = rand_img.shape[1:]

    def frame_args(self, duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }
        
    def plot_3d(self):
        # Define frames
        nb_frames = self.rand_img.shape[0]

        fig = go.Figure(frames=[go.Frame(data=go.Surface(
            z=(nb_frames*.1 - k * 0.1) * np.ones((self.r, self.c)),
            surfacecolor=np.flipud(self.rand_img[(nb_frames-1) - k]),
            ),
            name=str(k) # you need to name the frame for the animation to behave properly
            )
            for k in range(nb_frames)])
        
        fig.add_trace(go.Surface(
            z=nb_frames*.1 * np.ones((self.r, self.c)),
            surfacecolor=np.flipud(self.rand_img[(nb_frames-1)]),
            colorbar=dict(thickness=20, ticklen=4)
            ))

        sliders = [
                    {
                        "pad": {"b": 10, "t": 60},
                        "len": 0.9,
                        "x": 0.1,
                        "y": 0,
                        "steps": [
                            {
                                "args": [[f.name], self.frame_args(0)],
                                "label": str(k),
                                "method": "animate",
                            }
                            for k, f in enumerate(fig.frames)
                        ],
                    }
                ]

        fig.update_layout(
                title='Slices in volumetric data',
                width=600,
                height=600,
                scene=dict(
                        zaxis=dict(range=[-0.1, nb_frames*.1], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        ),
                updatemenus = [
                    {
                        "buttons": [
                            {
                                "args": [None, self.frame_args(100)],
                                "label": "&#9654;", # play symbol
                                "method": "animate",
                            },
                            {
                                "args": [[None], self.frame_args(0)],
                                "label": "&#9724;", # pause symbol
                                "method": "animate",
                            },
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 70},
                        "type": "buttons",
                        "x": 0.1,
                        "y": 0,
                    }
                ],
                sliders=sliders
        )
        fig.show()