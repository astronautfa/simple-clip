# Import required libraries
import torch  # PyTorch for deep learning
import torchvision.transforms as transforms  # For image transformations
import numpy as np  # For numerical operations
from collections import defaultdict  # For grouping data
from tqdm.auto import tqdm  # For progress bars
from io import BytesIO  # For handling binary data
from base64 import b64decode  # For decoding base64 images
from PIL import Image  # For image processing

# Set random seed for reproducibility
np.random.seed(42)


def get_image_tranforms(image_size=(224, 224)):
    """
    Creates a set of transformations to apply to images:
    1. Resize images to specified size
    2. Convert grayscale images to RGB if needed
    3. Convert to PyTorch tensor
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.Lambda(_grayscale_to_rgb),
        transforms.ToTensor()
    ])


def _grayscale_to_rgb(img):
    """
    Helper function to convert grayscale images to RGB format
    Only converts if the image isn't already in RGB mode
    """
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


class COCODataset(torch.utils.data.Dataset):
    """
    Dataset class for handling COCO-format data
    COCO is a large-scale object detection, segmentation, and captioning dataset
    """

    def __init__(self,
                 data,
                 tokenizer,
                 transforms,
                 image_key,
                 text_key,
                 shuffle_captions=True):
        """
        Initialize the dataset:
        - Processes all captions
        - Tokenizes them (converts text to numbers)
        - Groups images with their captions
        """
        # Process all captions from the data
        captions = []
        for row in tqdm(data):
            captions.append(row[text_key])

        print("captions", len(captions), captions[0])

        # Tokenize all captions (convert text to numbers that the model can understand)
        encoded_captions = tokenizer(captions,
                                     padding=True,
                                     truncation=True,
                                     max_length=100)

        # Group data by images (since one image can have multiple captions)
        grouped_data = defaultdict(list)
        for idx, row in enumerate(tqdm(data)):
            grouped_data[row[image_key][0]].append({
                "input_ids": encoded_captions["input_ids"][idx],
                "attention_mask": encoded_captions["attention_mask"][idx]
            })

        self.data = list(grouped_data.items())
        print("data", len(self.data))
        self.transforms = transforms
        self.shuffle_captions = shuffle_captions

    def __getitem__(self, idx):
        """
        Get a single item from the dataset:
        - Loads and processes the image
        - Gets corresponding caption
        - Returns both in the format needed for training
        """
        image_str, encoded_texts = self.data[idx]
        # Convert base64 string to image
        image = Image.open(BytesIO(b64decode(image_str)))
        # Apply image transformations
        image = self.transforms(image)

        # Either randomly select a caption or take the first one
        if self.shuffle_captions:
            encoded_text = np.random.choice(encoded_texts)
        else:
            encoded_text = encoded_texts[0]

        # Convert everything to PyTorch tensors
        instance = {
            key: torch.tensor(value)
            for key, value in encoded_text.items()
        }
        instance["image"] = image

        return instance

    def __len__(self):
        """Returns the total size of the dataset"""
        return len(self.data)


class SBUDataset(torch.utils.data.Dataset):
    """
    Dataset class for handling SBU (Scene Break Understanding) Captions dataset
    Similar to COCO but with a different data structure
    """

    def __init__(self,
                 data,
                 tokenizer,
                 transforms,
                 image_key="image_path",
                 text_key="captions",
                 shuffle_captions=True):
        """
        Initialize the dataset:
        - Filters out invalid entries
        - Processes and tokenizes captions
        """
        # Keep track of valid data entries and their captions
        valid_indices = []
        captions = []
        for idx, row in enumerate(tqdm(data)):
            if row[image_key]:  # Only keep entries with valid images
                captions.append(row[text_key])
                valid_indices.append(idx)

        print("captions", len(captions), captions[0])

        # Tokenize captions
        self.encoded_captions = tokenizer(captions,
                                          padding=True,
                                          truncation=True,
                                          max_length=100)

        # Select only valid data entries
        self.data = data.select(valid_indices)
        print("data", len(self.data),
              len(self.encoded_captions["input_ids"][0]))
        self.transforms = transforms
        self.shuffle_captions = shuffle_captions

    def __getitem__(self, idx):
        """
        Get a single item from the dataset
        Returns the processed image and its encoded caption
        """
        image = self.data[idx]["image"]
        image = self.transforms(image)

        instance = {
            key: torch.tensor(value[idx])
            for key, value in self.encoded_captions.items()
        }
        instance["image"] = image

        return instance

    def __len__(self):
        """Returns the total size of the dataset"""
        return len(self.data)


class CombinedDataset(torch.utils.data.Dataset):
    """
    Dataset class that combines multiple datasets (COCO, TextCap, and SBU Captions)
    Useful when training on multiple data sources
    """

    def __init__(self,
                 data,
                 tokenizer,
                 transforms,
                 shuffle_captions=True):
        """
        Initialize the combined dataset
        Simpler than other datasets as data is expected to be pre-processed
        """
        self.tokenizer = tokenizer
        self.data = data
        self.transforms = transforms
        self.shuffle_captions = shuffle_captions

    def __getitem__(self, idx):
        """
        Get a single item from the combined dataset
        Handles both image processing and caption selection
        """
        row = self.data[idx]
        image, captions = row["image"], row["caption"]

        # Choose a random caption or take the first one
        if self.shuffle_captions:
            caption = np.random.choice(captions)
        else:
            caption = captions[0]

        # Tokenize the selected caption
        encoded_caption = self.tokenizer(caption,
                                         padding="max_length",
                                         truncation=True,
                                         max_length=100)

        # Convert to PyTorch tensors
        instance = {
            key: torch.tensor(value)
            for key, value in encoded_caption.items()
        }
        instance["image"] = self.transforms(image)

        return instance

    def __len__(self):
        """Returns the total size of the dataset"""
        return len(self.data)
