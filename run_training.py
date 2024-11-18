import argparse  # Module for parsing command-line arguments
from train import train_clip  # Import the main training function

# Create argument parser object with description
parser = argparse.ArgumentParser(description='CLIP')

# Data-related arguments
parser.add_argument('--dataset_path',
                    default='./data',  # Default path to save datasets
                    help='Path where datasets will be saved')

parser.add_argument('--dataset_name',
                    default='coco',  # Default dataset
                    help='Dataset name',
                    # Allowed dataset options
                    choices=['textcap', 'coco', "sbucaptions", "combined", "yfcc7m"])

# Model architecture arguments
parser.add_argument(
    '--image_encoder_name',
    default='resnet50',  # Default image encoder
    # Available image encoder options
    choices=['resnet18', 'resnet50', "efficientnet"],
    help='image model architecture: resnet18, resnet50 or efficientnet (default: resnet50)'
)

parser.add_argument(
    '--text_encoder_name',
    default='distilbert-base-uncased',  # Default text encoder
    choices=['distilbert-base-uncased'],  # Available text encoder options
    help='text model architecture: distilbert-base-uncased (default: distilbert-base-uncased)'
)

# Model saving directory
parser.add_argument('-save_model_dir',
                    default='./models',
                    help='Path where models will be saved')

# Training hyperparameters
parser.add_argument('--num_epochs',
                    default=2,
                    type=int,  # Expects integer input
                    help='Number of epochs for training')

parser.add_argument('--image_size',
                    default=224,  # Standard image size for many vision models
                    type=int,
                    help='Image size')

parser.add_argument('-b',
                    '--batch_size',
                    default=256,
                    type=int,
                    help='Batch size')

parser.add_argument('-lr',
                    '--learning_rate',
                    default=1e-4,  # Default learning rate of 0.0001
                    type=float)

parser.add_argument('-wd',
                    '--weight_decay',
                    default=0.0,  # No weight decay by default
                    type=float)

# Training optimization options
parser.add_argument('--fp16_precision',
                    # Flag argument (True if specified, False otherwise)
                    action='store_true',
                    help='Whether to use 16-bit precision for GPU training')

# Evaluation settings
parser.add_argument('--imagenet_eval',
                    action='store_true',
                    help='Whether to evaluate on imagenet validation dataset. Required huggingface imagenet-1k dataset.')

parser.add_argument('--imagenet_eval_steps',
                    default=1000,
                    type=int,
                    help='Evaluate on imagenet every N steps')

# Logging and checkpointing
parser.add_argument('--log_every_n_steps',
                    default=50,
                    type=int,
                    help='Log every n steps')

parser.add_argument('--ckpt_path',
                    default=None,
                    type=str,
                    help='Specify path to clip_model.pth to resume training')

# Loss function selection
parser.add_argument('--use_siglip',
                    action='store_true',
                    help='Whether to use siglip loss')


def main():
    """
    Main function that:
    1. Parses command-line arguments
    2. Passes them to the training function
    """
    args = parser.parse_args()  # Parse command-line arguments
    train_clip(args)  # Start training with parsed arguments


# Standard Python idiom for running the main function
if __name__ == "__main__":
    main()

# Example usage (commented out):
"""
# Train with default parameters:
python script.py

# Train with custom parameters:
python script.py --dataset_name coco --image_encoder_name resnet50 --batch_size 128 --num_epochs 10 --fp16_precision

# Resume training from checkpoint:
python script.py --ckpt_path ./models/clip_model.pth --num_epochs 5
"""
