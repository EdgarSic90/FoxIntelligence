import pandas as pd

from tqdm import tqdm
from typing import Tuple
from image_hasher import ImageHasher


class DataProcessor:
    def __init__(self, external_file_path: str, orders_file_path: str, output_file_path: str, similarity_threshold: float = 0.95) -> None:
        """
        Initialize with paths to input datasets and output path.

        :param str external_file_path: Path to the external data CSV.
        :param str orders_file_path: Path to the orders data CSV.
        :param str output_file_path: Path to save processed data.
        :param float similarity_threshold: Image similarity threshold for deduplication.
        """
        self.external_file_path = external_file_path
        self.orders_file_path = orders_file_path
        self.output_file_path = output_file_path
        self.similarity_threshold = similarity_threshold
        self.image_hasher = ImageHasher()

    def remove_duplicate_images(self, df: pd.DataFrame, url_column: str, editor_column: str) -> pd.DataFrame:
        """
        Deduplicate images by similarity within each editor.

        :param pd.DataFrame df: DataFrame with image URLs and editor info.
        :param str url_column: Column with image URLs.
        :param str editor_column: Column with editor IDs.
        :return: DataFrame without duplicate images.
        :rtype: pd.DataFrame
        """
        unique_images, hash_dict = [], {}
        
        for editor, group in tqdm(df.groupby(editor_column), desc="Deduplicating images"):
            
            for idx, row in group.iterrows():
                img_hash = self.image_hasher.compute_image_hash(row[url_column])
                if img_hash and img_hash not in hash_dict:
                    hash_dict[img_hash] = (row['title'], row['editor'])
                    unique_images.append(row)
        
        return pd.DataFrame(unique_images)

    def process_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare data.

        :return: Tuple of orders and deduplicated external data.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        orders_df = pd.read_csv(self.orders_file_path)
        external_df = self.remove_duplicate_images(pd.read_csv(self.external_file_path), 'icon', 'editor')
        
        for col in ['image_matched_title', 'image_matched_editor', 'image_match_score',
                    'text_matched_title', 'text_matched_editor', 'text_match_score',
                    'final_title', 'final_editor']:
            orders_df[col] = None
        
        return orders_df, external_df

    def finalize_matches(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Finalize match results.

        :param pd.DataFrame orders_df: DataFrame with match data.
        :return: DataFrame with finalized matches.
        :rtype: pd.DataFrame
        """
        for index, row in orders_df.iterrows():
            text_title, image_title = row['text_matched_title'], row['image_matched_title']
            text_editor, image_editor = row['text_matched_editor'], row['image_matched_editor']
            
            if text_title == image_title and text_editor == image_editor:
                orders_df.at[index, 'final_title'] = text_title
                orders_df.at[index, 'final_editor'] = text_editor
            elif text_title and not image_title:
                orders_df.at[index, 'final_title'] = text_title
                orders_df.at[index, 'final_editor'] = text_editor
            elif image_title and not text_title:
                orders_df.at[index, 'final_title'] = image_title
                orders_df.at[index, 'final_editor'] = image_editor
        
        orders_df.to_csv(self.output_file_path, index=False)
        
        return orders_df
