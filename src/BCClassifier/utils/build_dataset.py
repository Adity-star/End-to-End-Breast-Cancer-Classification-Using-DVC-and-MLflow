import os
import shutil
import random
from imutils import paths
from BCClassifier import logger
from concurrent.futures import ThreadPoolExecutor # USed for faster data preparation
from tqdm import tqdm
import multiprocessing

def copy_file(args):
    """Helper function to copy a single file"""
    input_path, dest_path = args
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(input_path, dest_path)

def build_dataset(orig_input_dataset: str, train_path: str, val_path: str, test_path: str,
                  train_split: float = 0.8, val_split: float = 0.1, seed: int = 42):
    try:
        # Grab and shuffle image paths
        logger.info("Loading image paths...")
        image_paths = list(paths.list_images(orig_input_dataset))
        random.seed(seed)
        random.shuffle(image_paths)

        # Split into train and test
        i = int(len(image_paths) * train_split)
        train_paths = image_paths[:i]
        test_paths = image_paths[i:]

        # Split part of training into validation
        i = int(len(train_paths) * val_split)
        val_paths = train_paths[:i]
        train_paths = train_paths[i:]

        datasets = [
            ("training", train_paths, train_path),
            ("validation", val_paths, val_path),
            ("testing", test_paths, test_path),
        ]

        # Get number of CPU cores, but leave one free for system
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        
        for (dType, image_paths, base_output) in datasets:
            logger.info(f"Building '{dType}' split with {len(image_paths)} images")

            if not os.path.exists(base_output):
                logger.info(f"Creating '{base_output}' directory")
                os.makedirs(base_output)

            # Prepare copy operations
            copy_operations = []
            for input_path in image_paths:
                filename = os.path.basename(input_path)
                label = filename[-5:-4]  # e.g., '0' or '1'
                label_path = os.path.join(base_output, label)
                dest_path = os.path.join(label_path, filename)
                copy_operations.append((input_path, dest_path))

            # Use ThreadPoolExecutor for parallel file copying(makes process much faster)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                list(tqdm(
                    executor.map(copy_file, copy_operations),
                    total=len(copy_operations),
                    desc=f"Copying {dType} files"
                ))

        logger.info("Dataset building completed successfully.")

    except Exception as e:
        logger.error(f"Failed to build dataset: {e}")
        raise e
