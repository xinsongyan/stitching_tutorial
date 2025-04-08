from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from pathlib import Path

import cv2 as cv
import numpy as np

from stitching.images import Images
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher

def get_image_paths(img_set):
    return [str(path.relative_to('.')) for path in Path('imgs').rglob(f'{img_set}*')]

def plot_image(img, figsize_in_inches=(5, 5), figure_title=None):
    fig, ax = plt.subplots(figsize=figsize_in_inches, num=figure_title)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show(block=False)
    
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

def detect_features(imgs, detector='sift', debug=False):

    detector = FeatureDetector(detector=detector)
    features = [detector.detect_features(img) for img in imgs]
    keypoints_imgs = [detector.draw_keypoints(img, feature) for img, feature in zip(imgs, features)]
    if debug:
        plot_images(keypoints_imgs, (20, 10), 'Key Feature Points')
    return features

def match_features(imgs, features, debug=False):
    matcher = FeatureMatcher()
    matches = matcher.match_features(features)
    print("Confidence Matrix:\n", matcher.get_confidence_matrix(matches))

    if debug:
        # Convert the generator to a list
        all_relevant_matches = matcher.draw_matches_matrix(list(imgs), features, matches, conf_thresh=1, 
                                                                inliers=True, matchColor=(0, 255, 0))
        
        for idx1, idx2, img in all_relevant_matches:
            # print(f"Matches Image {idx1+1} to Image {idx2+1}")
            plot_image(img, (20,10), f'Matches Image {idx1+1} to Image {idx2+1}')
    return matches

def subset_images(images, features, matches, confidence_threshold=0.2):
    from stitching.subsetter import Subsetter

    subsetter = Subsetter(confidence_threshold=confidence_threshold)
    dot_notation = subsetter.get_matches_graph(images.names, matches)
    print(dot_notation)

    indices = subsetter.get_indices_to_keep(features, matches)

    images.subset(indices)
    features = subsetter.subset_list(features, indices)
    matches = subsetter.subset_matches(matches, indices)

    print(images.names)
    print("Confidence Matrix:\n", FeatureMatcher().get_confidence_matrix(matches))

def estimate_cameras(features, matches):
    from stitching.camera_estimator import CameraEstimator
    from stitching.camera_adjuster import CameraAdjuster
    from stitching.camera_wave_corrector import WaveCorrector

    camera_estimator = CameraEstimator()
    camera_adjuster = CameraAdjuster()
    wave_corrector = WaveCorrector()

    cameras = camera_estimator.estimate(features, matches)
    cameras = camera_adjuster.adjust(features, matches, cameras)
    cameras = wave_corrector.correct(cameras)
    return cameras


def warp_images(images, cameras, debug=False):
    from stitching.warper import Warper
    warper = Warper()
    warper.set_scale(cameras)

    final_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)
    warped_final_imgs = warper.warp_images(images, cameras)
    warped_final_masks = warper.create_and_warp_masks(final_sizes, cameras)
    final_corners, final_sizes = warper.warp_rois(final_sizes, cameras)
    print("Final Corners:", final_corners)
    print("Final Sizes:", final_sizes)
    if debug:
        warped_final_imgs = list(warped_final_imgs)
        warped_final_masks = list(warped_final_masks)
        fig, axs = plt.subplots(2, len(warped_final_imgs), figsize=(15, 10), num='Warped Images and Masks')
        for col, (img, mask) in enumerate(zip(warped_final_imgs, warped_final_masks)):
            axs[0, col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            axs[0, col].set_title(f'Image {col+1}', fontsize=8)
            axs[0, col].axis('off')
            axs[1, col].imshow(mask, cmap='gray')
            axs[1, col].set_title(f'Mask {col+1}', fontsize=8)
            axs[1, col].axis('off')

        plt.tight_layout()
        plt.show(block=False)

    return warped_final_imgs, warped_final_masks, final_corners, final_sizes


def timelapse_images(warped_final_imgs, final_corners, final_sizes, debug=False):
    from stitching.timelapser import Timelapser

    timelapser = Timelapser('as_is')
    timelapser.initialize(final_corners, final_sizes)

    if debug:
        fig, axs = plt.subplots(len(warped_final_imgs), 1, figsize=(10, 20), num='Timelapse Frames')
        for row, (img, corner) in enumerate(zip(warped_final_imgs, final_corners)):
            timelapser.process_frame(img, corner)
            frame = timelapser.get_frame()
            axs[row].imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            axs[row].set_title(f'Frame {row+1}', fontsize=8)
            axs[row].axis('off')
        plt.tight_layout()
        plt.show(block=False)


def crop_images(warped_final_imgs, warped_final_masks, final_corners, final_sizes, debug=False):
    from stitching.cropper import Cropper

    cropper = Cropper()
    mask = cropper.estimate_panorama_mask(warped_final_imgs, warped_final_masks, final_corners, final_sizes)
    lir = cropper.estimate_largest_interior_rectangle(mask)

    if debug: 
        plot = lir.draw_on(mask, size=2)
        plot_image(plot, (5,5), "Mask with LIR")

    cropper.prepare(warped_final_imgs, warped_final_masks, final_corners, final_sizes)
    cropped_final_masks = list(cropper.crop_images(warped_final_masks))
    cropped_final_imgs = list(cropper.crop_images(warped_final_imgs))
    final_corners, final_sizes = cropper.crop_rois(final_corners, final_sizes)

    if debug:
        from stitching.timelapser import Timelapser
        timelapser = Timelapser('as_is')
        timelapser.initialize(final_corners, final_sizes)

        fig, axs = plt.subplots(len(cropped_final_imgs), 1, figsize=(10, 20), num='Final Cropped Images')
        for row, (img, corner) in enumerate(zip(cropped_final_imgs, final_corners)):
            timelapser.process_frame(img, corner)
            frame = timelapser.get_frame()
            axs[row].imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            axs[row].set_title(f'Image {row+1}', fontsize=8)
            axs[row].axis('off')
        plt.tight_layout()
        plt.show(block=False)

    return cropped_final_imgs, cropped_final_masks, final_corners, final_sizes


def seam_images(cropped_imgs, cropped_masks, cropped_corners, cropped_sizes, debug=False):
    from stitching.seam_finder import SeamFinder

    seam_finder = SeamFinder()
    seam_masks = seam_finder.find(cropped_imgs, cropped_corners, cropped_masks)
    # seam_masks = [seam_finder.resize(seam_mask, mask) for seam_mask, mask in zip(seam_masks, cropped_masks)]

    seam_masks_plots = [SeamFinder.draw_seam_mask(img, seam_mask) for img, seam_mask in zip(cropped_imgs, seam_masks)]
    if debug:
        plot_images(seam_masks_plots, (15, 10), 'Seam Masks')
    return seam_masks

def blend_images(cropped_imgs, seam_masks, cropped_corners, cropped_sizes, debug=False):
    from stitching.blender import Blender

    blender = Blender()
    blender.prepare(cropped_corners, cropped_sizes)
    for img, mask, corner in zip(cropped_imgs, seam_masks, cropped_corners):
        blender.feed(img, mask, corner)
    panorama, _ = blender.blend()
    if debug:
        plot_image(panorama, (10, 10), 'Final Panorama')
    return panorama


def main1():
    img_paths = get_image_paths('fish')
    images = Images.of(img_paths)

    plot_images(list(images), (20, 20), 'Original Images')

    features = detect_features(images, detector='sift', debug=True)

    matches = match_features(images, features, debug=True)

    subset_images(images, features, matches, confidence_threshold=0.2)

    cameras = estimate_cameras(features, matches)

    warped_final_imgs, warped_final_masks, warped_corners, warped_sizes = warp_images(images, cameras, debug=True)

    timelapse_images(warped_final_imgs, warped_corners, warped_sizes)

    cropped_imgs, cropped_masks, cropped_corners, cropped_sizes = crop_images(warped_final_imgs, warped_final_masks, warped_corners, warped_sizes, debug=True)

    seam_masks = seam_images(cropped_imgs, cropped_masks, cropped_corners, cropped_sizes, debug=True)

    panorama = blend_images(cropped_imgs, seam_masks, cropped_corners, cropped_sizes, debug=True)

    plt.show(block=True)

def main2():
    from stitching import Stitcher

    img_paths = get_image_paths('fish')
    images = Images.of(img_paths)

    stitcher = Stitcher(detector='sift', confidence_threshold=0.2, matcher_type='affine')
    panorama = stitcher.stitch(img_paths)
    plot_image(panorama, (20,20))
    plt.savefig('panorama_stitcher.png', dpi=300)
    plt.show(block=True)

def main3():
    from stitching import AffineStitcher

    img_paths = get_image_paths('fish')
    images = Images.of(img_paths)

    settings = {
                "detector": 'sift',
                "confidence_threshold": 0.3,
                "crop": False,
                }    
    stitcher = AffineStitcher(**settings)
    panorama = stitcher.stitch(img_paths)

    plot_image(panorama, (20,20))
    plt.savefig('panorama_affine_stitcher.png', dpi=300)
    plt.show(block=True)

if __name__ == "__main__":
    # main1()
    # main2()
    main3()
