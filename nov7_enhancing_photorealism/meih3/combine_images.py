import cv2
import numpy as np
import os


def resize_with_aspect_ratio(image, scale=0.3):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def combine_images():
    # Define folders
    bg_folder = "./imgs/bg4"
    king_folder = "./imgs/king"
    output_folder = "./imgs/combined"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define file pairs with exact names
    file_pairs = {
        "albedo.png": "king_albedo.png",
        "irradiance.png": "king_irradiance.png",
        "metallic.png": "king_metallic.png",
        "normal.png": "king_normal.png",
        "roughness.png": "king_roughness.png",
    }

    for bg_file, king_file in file_pairs.items():
        try:
            # Construct full paths
            bg_path = os.path.join(bg_folder, bg_file)
            king_path = os.path.join(king_folder, king_file)

            print(f"Processing: {bg_file} + {king_file}")

            # Read images (read king image with alpha channel)
            bg_img = cv2.imread(bg_path)
            king_img = cv2.imread(
                king_path, cv2.IMREAD_UNCHANGED
            )  # Read with alpha channel

            if bg_img is None:
                print(f"Error: Could not load {bg_path}")
                continue
            if king_img is None:
                print(f"Error: Could not load {king_path}")
                continue

            # Resize king image while maintaining aspect ratio
            king_img_small = resize_with_aspect_ratio(king_img, scale=0.3)

            # Create a blank image the same size as bg_img
            temp_img = np.zeros_like(bg_img)

            # Calculate position to place the small king image
            y_offset = 200
            x_offset = 200

            # Get dimensions of the small king image
            h, w = king_img_small.shape[:2]

            # Extract alpha channel if it exists
            if king_img_small.shape[2] == 4:
                # Split into color and alpha channels
                king_rgb = king_img_small[:, :, :3]
                king_alpha = king_img_small[:, :, 3] / 255.0

                # Create ROI from bg image
                roi = bg_img[y_offset : y_offset + h, x_offset : x_offset + w]

                # Blend images based on alpha channel
                for c in range(3):
                    roi[:, :, c] = (
                        roi[:, :, c] * (1 - king_alpha) + king_rgb[:, :, c] * king_alpha
                    )

                # Place blended region back into bg image
                bg_img[y_offset : y_offset + h, x_offset : x_offset + w] = roi

                combined = bg_img
            else:
                # If no alpha channel, fall back to simple overlay
                temp_img[y_offset : y_offset + h, x_offset : x_offset + w] = (
                    king_img_small
                )
                combined = cv2.add(bg_img, temp_img)

            # Save combined image
            output_path = os.path.join(output_folder, bg_file)
            cv2.imwrite(output_path, combined)
            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Error processing {bg_file} and {king_file}: {str(e)}")
            continue


if __name__ == "__main__":
    print("Starting image combination process...")
    combine_images()
    print("Process completed!")
