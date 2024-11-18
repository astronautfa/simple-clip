# Import necessary modules
from torch import nn  # PyTorch neural network modules
# Hugging Face's DistilBERT implementation
from transformers import DistilBertModel, DistilBertConfig


class TextEncoder(nn.Module):
    """
    A neural network module that encodes text into fixed-size vectors using DistilBERT.

    DistilBERT is a smaller, faster version of BERT that retains most of its performance.
    It's been "distilled" from BERT, meaning it was trained to mimic BERT's behavior
    while using fewer parameters and computational resources.

    This encoder will:
    1. Take tokenized text as input
    2. Process it through DistilBERT
    3. Output a fixed-size vector representation of the text
    """

    def __init__(self, model_name, pretrained=True):
        """
        Initialize the text encoder.

        Args:
            model_name (str): The name/path of the DistilBERT model to use
                            (e.g., 'distilbert-base-uncased')
            pretrained (bool): Whether to use pre-trained weights or initialize randomly
                             Default: True (use pre-trained weights)
        """
        # Initialize the parent class (nn.Module)
        super(TextEncoder, self).__init__()

        if pretrained:
            # Load a pre-trained DistilBERT model with its weights
            # This is the most common usage as pre-trained models perform better
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            # Create a new DistilBERT model with default configuration
            # This would have random weights and need training from scratch
            config = DistilBertConfig()
            self.model = DistilBertModel(config)

    def forward(self, input_ids, attention_mask):
        """
        Process text through the encoder.

        This method:
        1. Passes the tokenized text through DistilBERT
        2. Gets the contextual embeddings for each token
        3. Averages these embeddings to get a single vector per text

        Args:
            input_ids (torch.Tensor): The tokenized text
                Shape: (batch_size, sequence_length)
                Each value is an integer representing a token

            attention_mask (torch.Tensor): Indicates which tokens are padding
                Shape: (batch_size, sequence_length)
                1 for real tokens, 0 for padding tokens

        Returns:
            torch.Tensor: The encoded text
                Shape: (batch_size, hidden_size)
                Usually hidden_size = 768 for base DistilBERT
        """
        # Process through DistilBERT
        # output is a dataclass containing:
        # - last_hidden_state: token embeddings for each position
        # - other optional outputs depending on configuration
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Get the contextual embeddings for each token
        # Shape: (batch_size, sequence_length, hidden_size)
        last_hidden_state = output.last_hidden_state

        # Average over the sequence length (dimension 1) to get a single vector per text
        # This performs mean pooling over all token embeddings
        # Shape: (batch_size, hidden_size)
        return last_hidden_state.mean(dim=1)


# Usage example (commented out):
"""
# Initialize the encoder
encoder = TextEncoder('distilbert-base-uncased')

# Assuming you have tokenized input:
# input_ids shape: (batch_size, sequence_length)
# attention_mask shape: (batch_size, sequence_length)
encoded_text = encoder(input_ids, attention_mask)
# encoded_text shape: (batch_size, 768)
"""
