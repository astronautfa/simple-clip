import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score


class ImageNetValidation:
    def __init__(self, transform):
        # Set device (GPU if available, else CPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load ImageNet validation dataset
        data = datasets.load_dataset("imagenet-1k")
        # Create custom dataset with transforms
        dataset = ImageNetDataset(data["validation"], transform)

        # Create dataloader with batch size 256 and 4 worker processes
        self.dataloader = DataLoader(dataset,
                                     batch_size=256,
                                     num_workers=4)

        # Convert numeric labels to text descriptions
        # Gets list of 1000 class names from ImageNet
        labels = data["validation"].features["label"].int2str(
            list(range(1000)))
        # Create text queries for zero-shot classification
        # Example: "a photo of a golden retriever"
        self.label_queries = [f"a photo of a {l}" for l in labels]

        # Initialize tokenizer and encode all text queries
        tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased")
        self.encoded_texts = tokenizer(
            self.label_queries, padding=True, truncation=True, max_length=100)

    @torch.inference_mode()
    def evaluate(self, model):
        # Main evaluation function
        model.eval()
        with torch.no_grad():
            # Get embeddings for all validation images and their labels
            # image_features shape: [num_val_images, embedding_dim]
            # labels shape: [num_val_images]
            image_features, labels = self._get_image_embs_labels(model)

            # Get embeddings for all class descriptions
            # text_features shape: [1000, embedding_dim]
            text_features = self._get_text_embs(model)

            # Compute similarity matrix between images and text descriptions
            # preds shape: [num_val_images, 1000]
            preds = image_features @ text_features.t()

            # Convert to CPU and get predictions
            labels = labels.cpu().detach().tolist()
            preds = preds.argmax(dim=-1).cpu().detach().tolist()

            # Calculate and print accuracy
            acc = accuracy_score(labels, preds)
            print("Accuracy ImagNet Val: ", acc)

    def _get_image_embs_labels(self, model):
        # Get embeddings for all validation images
        embs, labels = [], []
        for images, targets in tqdm(self.dataloader):
            with torch.no_grad():
                # Process batch of images
                # images shape: [batch_size, channels, height, width]
                images = images.to(self.device)
                # Get embeddings, shape: [batch_size, embedding_dim]
                out = model.extract_image_features(images)

                # Store embeddings and labels
                features = out.cpu().detach().tolist()
                embs.extend(features)
                labels.extend(targets.cpu().detach().tolist())

        # Return as tensors on device
        return torch.tensor(embs).to(self.device), torch.tensor(labels).to(self.device)

    def _get_text_embs(self, model):
        # Get embeddings for all class descriptions
        # Move encoded text tensors to device
        # Shape: [num_classes, max_length]
        input_ids = torch.tensor(
            self.encoded_texts["input_ids"]).to(self.device)
        attention_mask = torch.tensor(
            self.encoded_texts["attention_mask"]).to(self.device)

        # Return text embeddings, shape: [num_classes, embedding_dim]
        return model.extract_text_features(input_ids, attention_mask)


class ImageNetDataset(Dataset):
    def __init__(self,
                 data,
                 transforms):
        self.data = data  # HuggingFace dataset
        self.transforms = transforms  # Image transformations

    def __getitem__(self, idx):
        # Get single item from dataset
        image = self.data[idx]["image"]  # PIL Image
        label = self.data[idx]["label"]  # Integer label

        # Apply transforms and return
        image = self.transforms(image)  # Transform to tensor
        return image, torch.tensor(label)

    def __len__(self):
        return len(self.data)
