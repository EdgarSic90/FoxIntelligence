import requests
import imagehash

from tqdm import tqdm
from time import sleep
from io import BytesIO
from PIL import Image, ImageOps
from typing import List, Optional, Tuple
from requests.exceptions import RequestException
from concurrent.futures import ThreadPoolExecutor


class ImageHasher:
    
    def __init__(self, size: Tuple[int, int] = (100, 100), max_retries: int = 3, backoff_factor: int = 1) -> None:
        """
        Initialize the ImageHasher with parameters for image preprocessing and retry logic.

        :param Tuple[int, int] size: Target size for image resizing. Default is (100, 100).
        :param int max_retries: Maximum number of retries for failed image requests. Default is 3.
        :param int backoff_factor: Factor for exponential backoff in retry attempts. Default is 1.
        """
        self.size = size
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def preprocess_image(self, img: Image.Image) -> Image.Image:
        """
        Preprocess the image by converting to grayscale and resizing.

        :param Image.Image img: The image to preprocess.
        :return: The preprocessed image.
        :rtype: Image.Image
        """
        img = ImageOps.grayscale(img)
        img = img.resize(self.size, Image.LANCZOS)
        
        return img

    def compute_image_hash(self, url: str) -> Optional[imagehash.ImageHash]:
        """
        Compute the perceptual hash of an image from a URL with retry logic.

        :param str url: The URL of the image to hash.
        :return: The perceptual hash of the image, or None if failed.
        :rtype: Optional[imagehash.ImageHash]
        """
        for attempt in range(self.max_retries):
            
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    img = self.preprocess_image(img)
                    return imagehash.phash(img)
            except RequestException as e:
                print(f"Error processing image from {url}: {e}")
                sleep(self.backoff_factor * (2 ** attempt))  # Exponential backoff
       
        print(f"Failed to process image from {url} after {self.max_retries} attempts")
        
        return None

    def compute_hashes_in_parallel(self, urls: List[str]) -> List[Optional[imagehash.ImageHash]]:
        """
        Compute image hashes for a list of URLs in parallel.

        :param List[str] urls: List of image URLs to process.
        :return: List of image hashes corresponding to the input URLs.
        :rtype: List[Optional[imagehash.ImageHash]]
        """
        with ThreadPoolExecutor() as executor:
            hashes = list(tqdm(executor.map(self.compute_image_hash, urls), total=len(urls), desc="Processing images"))
        
        return hashes
