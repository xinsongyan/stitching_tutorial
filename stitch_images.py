from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from pathlib import Path

import cv2 as cv
import numpy as np


from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher

def get_image_paths(img_set):
    return [str(path.relative_to('.')) for path in Path('imgs').rglob(f'{img_set}*')]

def plot_image(img, figsize_in_inches=(5,5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()
    
def plot_images(imgs, figsize_in_inches=(5, 5), figure_title=None):
    """
    Plots images based on input type (image arrays or file paths).

    Args:
        imgs (list): List of image arrays or file paths.
        figsize_in_inches (tuple): Size of the figure.
        figure_title (str): Title for the entire figure.
    """

    # Check if the input is a list of file paths (strings)
    if isinstance(imgs[0], str):
        # Load images from file paths
        loaded_imgs = [cv.imread(img_path) for img_path in imgs]
        fig, axs = plt.subplots(1, len(loaded_imgs), figsize=figsize_in_inches, num=figure_title)
        for col, (img, img_path) in enumerate(zip(loaded_imgs, imgs)):
            axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            axs[col].set_title(Path(img_path).name, fontsize=8)
            axs[col].axis('off')
    else:
        # Assume the input is a list of image arrays
        fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches, num=figure_title)
        for col, img in enumerate(imgs):
            axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            axs[col].axis('off')



    plt.tight_layout()
    plt.show(block=False)

def resize_images(imgs):
    medium_imgs = list(images.resize(Images.Resolution.MEDIUM))
    low_imgs = list(images.resize(Images.Resolution.LOW))
    final_imgs = list(images.resize(Images.Resolution.FINAL))
    return medium_imgs, low_imgs, final_imgs

def detect_features(imgs, debug=False):

    detector = FeatureDetector()
    features = [detector.detect_features(img) for img in imgs]
    keypoints_imgs = [detector.draw_keypoints(img, feature) for img, feature in zip(imgs, features)]
    if debug:
        plot_images(keypoints_imgs, (20, 10), 'Key Feature Points')
    return features

def match_features(imgs, features, debug=False):
    matcher = FeatureMatcher()
    matches = matcher.match_features(features)
    matcher.get_confidence_matrix(matches)
    if debug:
        # Convert the generator to a list
        all_relevant_matches = list(matcher.draw_matches_matrix(imgs, features, matches, conf_thresh=1, 
                                                                inliers=True, matchColor=(0, 255, 0)))
        fig, axs = plt.subplots(len(all_relevant_matches), 1, figsize=(20, 10 * len(all_relevant_matches)))
        for i, (idx1, idx2, img) in enumerate(all_relevant_matches):
            axs[i].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            axs[i].text(
                0.5, 0.95, f"Matches Image {idx1+1} to Image {idx2+1}",
                fontsize=12, color='black', ha='center', va='bottom', transform=axs[i].transAxes,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )
            axs[i].axis('off')
        plt.tight_layout()
        plt.show(block=False)

if __name__ == "__main__":
    img_paths = get_image_paths('weir')
    imgs = [cv.imread(img_path) for img_path in img_paths]

    plot_images(img_paths, (20, 10), 'Original Images')

    features = detect_features(imgs, debug=True)

    match_features(imgs, features, debug=True)


    plt.show(block=True)