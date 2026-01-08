# augment_handpd.py

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from pathlib import Path

# -------------------- CONFIG --------------------
INPUT_DIR = "/Users/manjusha/Documents/research/dataset"                 # your original dataset root
OUTPUT_DIR = "/Users/manjusha/Documents/research/augmented_dataset"      # destination for augmented images
AUG_PER_IMAGE = 5                     # how many augmented images to create per original
TARGET_SIZE = (224, 224)              # resize images (common for CNNs); set to your model's expected size
ALLOWED_EXTS = { ".jpg"}  # file extensions to process
# ------------------------------------------------

# Augmentation parameters: tune these as needed
datagen = ImageDataGenerator(
    rotation_range=15,            # rotate up to 15 degrees
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.08,
    zoom_range=0.08,
    brightness_range=(0.85, 1.15),
    horizontal_flip=True,
    fill_mode='nearest'
)

def make_output_path(inp_root, out_root, rel_path):
    out_path = Path(out_root) / rel_path
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path

def augment_folder(input_subfolder: Path, output_subfolder: Path):
    for img_file in input_subfolder.iterdir():
        if img_file.is_file() and img_file.suffix.lower() in ALLOWED_EXTS:
            try:
                # load and optionally resize
                img = load_img(img_file, target_size=TARGET_SIZE)  # PIL image
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)  # (1, H, W, C)

                # Save original also to output folder (optional — comment out if not needed)
                original_save_path = output_subfolder / f"orig_{img_file.stem}{img_file.suffix}"
                if not original_save_path.exists():
                    img.save(original_save_path)

                # generate augmented images
                i = 0
                for batch in datagen.flow(x, batch_size=1):
                    i += 1
                    aug_name = output_subfolder / f"aug_{img_file.stem}_{i}{img_file.suffix}"
                    # keras flow yields numpy array; we can convert back to PIL via keras' array_to_img
                    from tensorflow.keras.preprocessing.image import array_to_img
                    array_to_img(batch[0]).save(aug_name)
                    if i >= AUG_PER_IMAGE:
                        break
            except Exception as e:
                print(f"Skipping {img_file} due to error: {e}")

def main():
    input_root = Path(INPUT_DIR)
    output_root = Path(OUTPUT_DIR)
    if not input_root.exists():
        raise SystemExit(f"Input directory '{INPUT_DIR}' not found.")

    # Walk through classes and subfolders
    for class_dir in input_root.iterdir():             # healthy, parkinson
        if not class_dir.is_dir():
            continue
        for subtype_dir in class_dir.iterdir():       # circle, meander, spiral
            if not subtype_dir.is_dir():
                continue
            # prepare corresponding output folder
            rel_path = subtype_dir.relative_to(input_root)
            out_subfolder = make_output_path(input_root, output_root, rel_path)
            print(f"Augmenting {subtype_dir} -> {out_subfolder}")
            augment_folder(subtype_dir, out_subfolder)

    print("Augmentation finished. Augmented dataset in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
