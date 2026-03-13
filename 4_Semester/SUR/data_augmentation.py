import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import shutil


def rotate_image(image, angle):
    """Rotate the image"""
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

def add_gaussian_noise(image, mean=0, std=25):
    """Add Gaussian noise"""
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def adjust_brightness(image, factor=1.5):
    """Adjust image brightness."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def augment_images(input_root, output_root, augmentations):
    # Create output directory structure
    for class_dir in os.listdir(input_root):
        class_path = os.path.join(input_root, class_dir)
        if os.path.isdir(class_path):
            os.makedirs(os.path.join(output_root, class_dir), exist_ok=True)

    # Process each class folder
    for class_dir in tqdm(os.listdir(input_root)):
        class_path = os.path.join(input_root, class_dir)
        if not os.path.isdir(class_path):
            continue

        output_class_path = os.path.join(output_root, class_dir)

        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_path, filename)
                image = cv2.imread(image_path)

                # Save original image first
                shutil.copy2(image_path, os.path.join(output_class_path, filename))

                # Apply selected augmentations
                if 'rotate' in augmentations:
                    for angle in [15, 30, -15, -30]:
                        rotated = rotate_image(image, angle)
                        new_name = f"{os.path.splitext(filename)[0]}_rot{angle}.png"
                        cv2.imwrite(os.path.join(output_class_path, new_name), rotated)

                if 'noise' in augmentations:
                    noisy = add_gaussian_noise(image)
                    new_name = f"{os.path.splitext(filename)[0]}_noise.png"
                    cv2.imwrite(os.path.join(output_class_path, new_name), noisy)

                if 'brightness' in augmentations:
                    for factor in [0.7, 1.3]:
                        adjusted = adjust_brightness(image, factor)
                        new_name = f"{os.path.splitext(filename)[0]}_bright{factor}.png"
                        cv2.imwrite(os.path.join(output_class_path, new_name), adjusted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment images")
    parser.add_argument("input_root", type=str, help="Root directory containing class folders (1-31)")
    parser.add_argument("output_root", type=str, help="Output directory for augmented images")
    parser.add_argument("--rotate", action="store_true", help="Apply rotation augmentation")
    parser.add_argument("--noise", action="store_true", help="Apply Gaussian noise augmentation")
    parser.add_argument("--brightness", action="store_true", help="Apply brightness adjustment")
    args = parser.parse_args()

    # Determine which augmentations to apply
    selected_augmentations = []
    if args.rotate:
        selected_augmentations.append('rotate')
    if args.noise:
        selected_augmentations.append('noise')
    if args.brightness:
        selected_augmentations.append('brightness')

    print(f"Applying augmentations: {', '.join(selected_augmentations) or 'None'}")
    augment_images(args.input_root, args.output_root, selected_augmentations)
    print("Augmentation complete!")
