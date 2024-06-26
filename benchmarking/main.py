import os
import numpy as np
from PIL import Image

from transformer import Transformer
from blockhash import blockhash
from neuralhash import neuralhash

dataset_folder = '/dataset'
image_files = [f for f in os.listdir(dataset_folder)]
images = [Image.open(os.path.join(dataset_folder, image_file)) for image_file in image_files]

transformations = ['jpeg', 'crop', 'screenshot', 'double screenshot']
hash_methods = [blockhash, neuralhash]

original_hashes_per_method = [hash_method(images) for hash_method in hash_methods]
bit_match_percentage = np.zeros((len(hash_methods), len(transformations), len(image_files)))

for i, (image, original_hashes) in enumerate(zip(images, original_hashes_per_method)):
    original_hashes = [hash_method(image) for hash_method in hash_methods]

    transformed_images = []
    for j, transformation in enumerate(transformations):
        transformed_image = Transformer().transform(image, transformation)
        transformed_images.append(transformed_image)

    for i, (original_hash, hash_method) in enumerate(zip(original_hashes, hash_methods)):

        modified_hashes = hash_method(transformed_images)
        bit_overlap_percentages = [sum(c1 == c2 for c1, c2 in zip(modified_hash, original_hash)) / len(modified_hash) for modified_hash in modified_hashes]

        bit_match_percentage[i][j] = bit_overlap_percentages

for i, image_file in enumerate(image_files):
    print(f"Image: {image_file}")
    for j, transformation in enumerate(transformations):
        print(f"Transformation: {transformation}, Bit Match Percentage: {bit_match_percentage[i][j]}")