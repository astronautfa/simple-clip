import argparse

from train import train_clip

parser = argparse.ArgumentParser(description='ReLIC')
parser.add_argument('--dataset_path',
                    default='./data',
                    help='Path where datasets will be saved')
parser.add_argument('--dataset_name',
                    default='coco',
                    help='Dataset name',
                    choices=['textcap', 'coco', "sbucaptions"])
parser.add_argument(
    '--image_encoder_name',
    default='resnet50',
    choices=['resnet18', 'resnet50', "efficientnet"],
    help=
    'image model architecture: resnet18, resnet50 or efficientnet (default: resnet50)'
)
parser.add_argument(
    '--text_encoder_name',
    default='distilbert-base-uncased',
    choices=['distilbert-base-uncased'],
    help=
    'text model architecture: distilbert-base-uncased (default: distilbert-base-uncased)'
)
parser.add_argument('-save_model_dir',
                    default='./models',
                    help='Path where models')
parser.add_argument('--num_epochs',
                    default=100,
                    type=int,
                    help='Number of epochs for training')
parser.add_argument('--image_size', default=224, type=int, help='Image size')
parser.add_argument('-b',
                    '--batch_size',
                    default=256,
                    type=int,
                    help='Batch size')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float)
parser.add_argument('--fp16_precision',
                    action='store_true',
                    help='Whether to use 16-bit precision for GPU training')
parser.add_argument('--log_every_n_steps',
                    default=50,
                    type=int,
                    help='Log every n steps')
parser.add_argument('--ckpt_path',
                    default=None,
                    type=str,
                    help='Specify path to relic_model.pth to resume training')


def main():
    args = parser.parse_args()
    train_clip(args)


if __name__ == "__main__":
    main()
