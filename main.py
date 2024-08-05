from data_processor import DataProcessor
from text_matcher import TextMatcher
from image_matcher import ImageMatcher

class DataProcessor:
    def __init__(self, external_file_path: str, orders_file_path: str, output_file_path: str) -> None:
        """
        Initializes the DataProcessor with paths to input datasets and an output path for saving the processed data.

        :param str external_file_path: Path to the CSV file containing external data about apps.
        :param str orders_file_path: Path to the CSV file containing data on user orders.
        :param str output_file_path: Path to save the CSV file with enriched and finalized data.
        """
        self.external_file_path = external_file_path
        self.orders_file_path = orders_file_path
        self.output_file_path = output_file_path


    def launch(self)-> None:
        """
        Executes the complete data processing workflow, including data loading, processing, matching, and saving.

        This method integrates several steps into a cohesive process:
        - Loads data from specified CSV files.
        - Applies text-based and image-based matching algorithms to enrich the data.
        - Finalizes the matches to consolidate and validate the results.
        - Saves the enriched dataset to a specified output file.
        """
            # Load and process initial data
        orders_df, external_df = self.process_data()

            # Text-based matching
        text_matcher = TextMatcher()
        orders_df = text_matcher.match_titles_with_products(orders_df, external_df)

            # Image-based matching
        image_matcher = ImageMatcher()
        orders_df = image_matcher.match_images(orders_df, external_df)

            # Aggregate results
        orders_df = self.finalize_matches(orders_df)

            # Save the processed data to a file
        orders_df.to_csv(self.output_file_path, index=False)

def main():
   
    # to update
    external_file_path = '/Users/edgarsicat/Documents/Data Projects/Foxintelligence/DNV Technical Test - Data Version/technical_test_external_source_extract.csv'
    orders_file_path = '/Users/edgarsicat/Documents/Data Projects/Foxintelligence/DNV Technical Test - Data Version/technical_test_table_extract.csv'
    output_file_path = '/Users/edgarsicat/Documents/Data Projects/Foxintelligence/process_data/enriched_orders.csv'

    
    data_processor = DataProcessor(external_file_path, orders_file_path, output_file_path)
    data_processor.launch()

if __name__ == '__main__':
    main()
