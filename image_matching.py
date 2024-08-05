import concurrent.futures
import requests
import imagehash
from PIL import Image, ImageOps, ImageFilter
import pandas as pd
from io import BytesIO
from tqdm import tqdm

class ImageMatcher:
    def __init__(self, orders_csv_path, external_csv_path, output_file_path):
        self.orders_csv_path = orders_csv_path
        self.external_csv_path = external_csv_path
        self.output_file_path = output_file_path

    def preprocess_image(self, img):
        # Convert to grayscale
        img = ImageOps.grayscale(img)
        # Resize to a standard size using LANCZOS filter
        img = img.resize((32, 32), Image.LANCZOS)  # Smaller size for faster processing
        return img

    def get_image_hash_from_url(self, url):
        try:
            response = requests.get(url, stream=True, timeout=5)  # Set a timeout to avoid long waits
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img = self.preprocess_image(img)
            return imagehash.phash(img)  # Use pHash for perceptual hashing
        except Exception as e:
            print(f"Error processing image from {url}: {e}")
            return None

    def compute_hashes_parallel(self, df, url_column, key_column_func):
        hashes = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_key = {
                executor.submit(self.get_image_hash_from_url, row[url_column]): key_column_func(row)
                for _, row in df.iterrows()
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_key), total=len(future_to_key), desc="Hashing images"):
                key = future_to_key[future]
                try:
                    img_hash = future.result()
                    if img_hash is not None:
                        hashes[key] = img_hash
                except Exception as e:
                    print(f"Hash computation failed for {key}: {e}")
        return hashes

    def match_images(self):
        # Load external data CSV
        external_df = pd.read_csv(self.external_csv_path)

        # Compute hashes for external images in parallel
        external_hashes = self.compute_hashes_parallel(
            external_df,
            'icon',  # Assuming 'logo_url' is the column for image URLs
            lambda row: f"{row['title']}_{row['editor']}"
        )

        # Load orders CSV
        orders_df = pd.read_csv(self.orders_csv_path)

        # Compute hashes for order images in parallel
        order_hashes = self.compute_hashes_parallel(
            orders_df,
            'product_url_img',  # Assuming 'logo_url' is the column for image URLs
            lambda row: row['id_order']  # Assuming 'id_order' is the column for order identifiers
        )

        # Match order images to external images using hash map
        matched_titles = []
        for order_id, order_hash in order_hashes.items():
            best_match = None
            best_editor = None
            min_distance = float('inf')
            for title_editor, ext_hash in external_hashes.items():
                distance = order_hash - ext_hash
                if distance < min_distance:
                    min_distance = distance
                    best_match, best_editor = title_editor.rsplit('_', 1)
                # Early exit if a perfect match is found
                if min_distance == 0:
                    break
            matched_titles.append((best_match, best_editor, min_distance))

        # Add matched image info to the DataFrame
        orders_df['img_matched_title'], orders_df['img_matched_editor'], orders_df['img_matching_score'] = zip(*matched_titles)

        # Save the enriched CSV with image matching info
        orders_df.to_csv(self.output_file_path, index=False)
