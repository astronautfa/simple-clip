import torch
from transformers import DistilBertTokenizer
import datasets
import webdataset as wds
from simple_clip.encoders import resnet18, resnet50, efficientnet_v2_s, TextEncoder
from simple_clip.custom_datasets.clip_datasets import COCODataset, SBUDataset, CombinedDataset


def get_dataset(dataset_name, dataset_path, transforms, split="train", shuffle_captions=True):
    # Initialize tokenizer for text processing
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # COCO dataset handler
    if dataset_name == "coco":
        data = datasets.load_dataset("MMInstruction/M3IT", dataset_name)[split]
        return COCODataset(data,
                           tokenizer,
                           transforms=transforms,
                           image_key="image_base64_str",
                           text_key="outputs",
                           shuffle_captions=shuffle_captions)

    # TextCap dataset handler
    elif dataset_name == "textcap":
        data = datasets.load_dataset("MMInstruction/M3IT", dataset_name)[split]
        return COCODataset(data,
                           tokenizer,
                           transforms=transforms,
                           image_key="image_base64_str",
                           text_key="outputs",
                           shuffle_captions=shuffle_captions)

    # SBU Captions dataset handler
    elif dataset_name == "sbucaptions":
        data = datasets.load_from_disk(
            f"{dataset_path}/sbu_captions_images")["train"]
        return SBUDataset(data,
                          tokenizer,
                          transforms=transforms,
                          image_key="image",
                          text_key="caption")

    # Combined dataset handler (COCO + TextCap + SBU)
    elif dataset_name == "combined":
        # Load and concatenate multiple datasets
        data_coco = datasets.load_from_disk(f"{dataset_path}/coco")["train"]
        data_textcap = datasets.load_from_disk(
            f"{dataset_path}/textcap")["train"]
        data_sbu = datasets.load_from_disk(
            f"{dataset_path}/sbu_captions_images")
        # Ensure consistent format for captions
        data_sbu = data_sbu.map(
            lambda example: {"caption": [example["caption"]]})
        data = datasets.concatenate_datasets(
            [data_coco, data_textcap, data_sbu])
        return CombinedDataset(data, tokenizer, transforms=transforms)

    # YFCC7M dataset handler using WebDataset format
    elif dataset_name == "yfcc7m":
        def prepare_data(x):
            # Process each instance from tar files
            caption = x["txt"]
            encoded_caption = tokenizer(caption,
                                        padding="max_length",
                                        truncation=True,
                                        max_length=100)
            image = transforms(x["jpg"])

            # Convert to tensor format
            instance = {
                key: torch.tensor(value)
                for key, value in encoded_caption.items()
            }
            instance["image"] = image
            return instance

        # Create WebDataset from tar files with shuffling
        dataset = wds.WebDataset([f"{dataset_path}/yfcc7m/{i:05d}.tar" for i in range(1538)],
                                 shardshuffle=True,
                                 cache_dir=f"{dataset_path}/yfcc7m_training_cache")
        dataset = dataset.shuffle(1000, initial=100).decode(
            "pil").map(prepare_data)
        return dataset

    raise Exception(f"Invalid dataset name {
                    dataset_name} - options are [coco, sbucaptions, combined, yfcc7m]")


def get_image_encoder(model_name):
    # Factory function for image encoders
    if model_name == "resnet18":
        return resnet18()
    elif model_name == "resnet50":
        return resnet50()
    elif model_name == "efficientnet":
        return efficientnet_v2_s()
    raise Exception(
        "Invalid model name - options are [resnet18, resnet50, efficientnet]")


def get_text_encoder(model_name="distilbert-base-uncased"):
    # Create text encoder instance (DistilBERT by default)
    return TextEncoder(model_name)


def accuracy(output, target, topk=(1,)):
    # Calculate top-k accuracy metrics
    # output shape: [batch_size, num_classes]
    # target shape: [batch_size]
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top k predictions
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # Calculate accuracy for each k
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.inference_mode
def get_feature_size(encoder):
    """Get output feature dimension from encoder using dummy input."""
    encoder.eval()
    # Use small input size for efficiency
    # [batch_size, channels, height, width]
    dummy_input = torch.randn(1, 3, 32, 32)
    output = encoder(dummy_input)
    return output.shape[1]  # Return feature dimension
