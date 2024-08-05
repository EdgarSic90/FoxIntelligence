import pandas as pd

from tqdm import tqdm
from fuzzywuzzy import process, fuzz


class TextMatcher:
    
    def __init__(self, text_similarity_threshold: float = 0.9) -> None:
        """
        Initialize the TextMatcher with a threshold for text similarity.

        :param float text_similarity_threshold: Minimum similarity score required to consider a match. Default is 0.9.
        """
        self.text_similarity_threshold = text_similarity_threshold

    def match_titles_with_products(self, orders_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
        """
        Match product names in orders with titles in external data using fuzzy matching.

        :param pd.DataFrame orders_df: DataFrame containing orders with product names to match.
        :param pd.DataFrame external_df: DataFrame containing titles and editors to match against.
        :return: Updated orders DataFrame with text match results.
        :rtype: pd.DataFrame
        """
        title_editor_dict = external_df.set_index('title')['editor'].to_dict()

        for i, product_row in tqdm(orders_df.iterrows(), total=orders_df.shape[0], desc="Matching titles with products"):
            product_name = product_row['product_name'].lower().strip()
            
            best_match = process.extractOne(product_name, title_editor_dict.keys(), scorer=fuzz.partial_ratio)
            if best_match:
                matched_title, score = best_match
                editor = title_editor_dict[matched_title]
                best_score = score / 100.0

                if best_score >= self.text_similarity_threshold:
                    orders_df.at[i, 'text_matched_title'] = matched_title
                    orders_df.at[i, 'text_matched_editor'] = editor
                    orders_df.at[i, 'text_match_score'] = best_score

        return orders_df
