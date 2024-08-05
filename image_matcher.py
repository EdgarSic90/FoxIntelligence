import pandas as pd

from image_hasher import ImageHasher


class ImageMatcher:
    
    def __init__(self, hash_diff_threshold: int = 5) -> None:
        """
        Initialize the ImageMatcher with a threshold for hash difference.

        :param int hash_diff_threshold: Maximum allowable difference between image hashes for a match. Default is 5.
        """
        self.image_hasher = ImageHasher()
        self.hash_diff_threshold = hash_diff_threshold

    def match_images(self, orders_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
        """
        Matches images from orders with external images based on perceptual hashing.

        :param pd.DataFrame orders_df: DataFrame containing orders with image URLs to match.
        :param pd.DataFrame external_df: DataFrame containing external image URLs and associated metadata.
        :return: Updated orders DataFrame with image match results.
        :rtype: pd.DataFrame
        """
        external_urls = external_df['icon'].tolist()
        external_hashes = self.image_hasher.compute_hashes_in_parallel(external_urls)
        external_df['image_hash'] = external_hashes

        # lookup dictionary 
        external_hash_dict = {
            ext_hash: (row['title'], row['editor'])
            for ext_hash, (_, row) in zip(external_hashes, external_df.iterrows())
            if ext_hash is not None
        }

        # Compute hashes
        order_urls = orders_df['product_url_img'].tolist()
        order_hashes = self.image_hasher.compute_hashes_in_parallel(order_urls)

        for index, (order_row, order_hash) in enumerate(zip(orders_df.iterrows(), order_hashes)):
            
            if order_hash is None:
                continue

            # Compare with external image hashes
            for ext_hash, (title, editor) in external_hash_dict.items():
                hash_diff = order_hash - ext_hash
                if hash_diff < self.hash_diff_threshold:
                    orders_df.at[index, 'image_matched_title'] = title
                    orders_df.at[index, 'image_matched_editor'] = editor
                    orders_df.at[index, 'image_match_score'] = 1 - hash_diff / len(ext_hash.hash.flatten())
                    break

        return orders_df
