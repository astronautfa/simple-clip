import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from simple_clip.utils import get_feature_size


def contrastive_loss(logits):
    # Input logits shape: [batch_size, batch_size] - similarity matrix between image and text embeddings
    # Each row represents one image's similarity with all texts
    targets = torch.arange(logits.size(0)).to(logits.device)
    # Creates target indices [0,1,2,...,batch_size-1] for matching positive pairs

    # Cross entropy loss for image-to-text direction
    # For each image (row), the target is matching text at same index
    loss_images = F.cross_entropy(logits, targets)

    # Cross entropy loss for text-to-image direction
    # Transpose logits to compute loss from text perspective
    loss_texts = F.cross_entropy(logits.t(), targets)

    # Average both directions for symmetric loss
    return (loss_images + loss_texts) / 2


def siglip_loss(logits):
    # Alternative loss function using sigmoid
    n = logits.size(0)  # batch size

    # Creates a matrix with 1s on diagonal (positive pairs) and -1s elsewhere (negative pairs)
    # Shape: [batch_size, batch_size]
    labels = 2 * torch.eye(n, device=logits.device) - 1

    # Applies sigmoid loss: -log(sigmoid(x)) for positive pairs
    # and -log(sigmoid(-x)) for negative pairs
    return -torch.sum(F.logsigmoid(labels * logits)) / n


class CLIP(torch.nn.Module):
    def __init__(self,
                 image_encoder,  # CNN or Vision Transformer
                 text_encoder,   # BERT or similar transformer
                 image_mlp_dim=False,  # Size of image encoder output
                 # Size of text encoder output (e.g., BERT hidden size)
                 text_mlp_dim=768,
                 proj_dim=256,         # Final projection dimension
                 init_tau=np.log(1.0),  # Initial temperature scaling
                 init_b=0):            # Initial bias term
        super(CLIP, self).__init__()

        # Get image encoder output size if not provided
        if not image_mlp_dim:
            image_mlp_dim = get_feature_size(image_encoder)

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # MLP for projecting image features
        # Shape: image_mlp_dim -> image_mlp_dim -> proj_dim
        self.image_projection = torch.nn.Sequential(
            torch.nn.Linear(image_mlp_dim, image_mlp_dim, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(image_mlp_dim, proj_dim, bias=False))

        # MLP for projecting text features
        # Shape: text_mlp_dim -> text_mlp_dim -> proj_dim
        self.text_projection = torch.nn.Sequential(
            torch.nn.Linear(text_mlp_dim, text_mlp_dim, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(text_mlp_dim, proj_dim, bias=False))

        # Learnable temperature parameter (scalar)
        self.t_prime = nn.Parameter(torch.ones([]) * init_tau)
        # Learnable bias parameter (scalar)
        self.b = nn.Parameter(torch.ones([]) * init_b)

    def forward(self, image, input_ids, attention_mask):
        # Extract features from both modalities
        # image shape: [batch_size, channels, height, width]
        # input_ids shape: [batch_size, sequence_length]
        # attention_mask shape: [batch_size, sequence_length]

        image_features = self.extract_image_features(image)
        text_features = self.extract_text_features(input_ids, attention_mask)

        # Normalize feature vectors to unit length
        # Features shape after projection: [batch_size, proj_dim]
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Compute similarity matrix with temperature scaling and bias
        # Output shape: [batch_size, batch_size]
        # Each entry (i,j) represents similarity between i-th image and j-th text
        return image_features @ text_features.t() * self.t_prime.exp() + self.b

    def extract_image_features(self, images):
        # Process images through encoder and projection
        # Input shape: [batch_size, channels, height, width]
        # Output shape: [batch_size, proj_dim]
        image_features = self.image_encoder(images)
        return self.image_projection(image_features)

    def extract_text_features(self, input_ids, attention_mask):
        # Process text through encoder and projection
        # Input shapes: [batch_size, sequence_length]
        # Output shape: [batch_size, proj_dim]
        text_features = self.text_encoder(input_ids, attention_mask)
        return self.text_projection(text_features)
