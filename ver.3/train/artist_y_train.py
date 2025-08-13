import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertLMHeadModel, BertTokenizer
from tqdm import tqdm
import warnings

# We need to import the ArtistX class to load its structure and weights
from artist_x_train import ArtistX 

warnings.filterwarnings("ignore")

# --- 1. Model Definition (ArtistY) ---
class ArtistY(nn.Module):
    """
    ArtistY: Inherits X's encoder but uses its own decoder
    to reinterpret the latent vector into new, creative text.
    """
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
        
        # Encoder: Same structure as X's. Will be overwritten with trained weights.
        self.encoder = BertModel.from_pretrained(model_name, local_files_only=True)
        
        # Decoder: A new, separate decoder for creative generation.
        self.decoder = BertLMHeadModel.from_pretrained(model_name, is_decoder=True, local_files_only=True)

    def load_encoder_from_x(self, x_model: ArtistX):
        """
        Loads the trained encoder weights from an ArtistX instance.
        """
        # Get the state dictionary from X's encoder
        encoder_weights = x_model.encoder.state_dict()
        # Load the weights into Y's encoder
        self.encoder.load_state_dict(encoder_weights)
        print("✅ Y has successfully inherited X's encoding ability.")

    def reinterpret(self, text: str, max_new_tokens=30) -> str:
        """
        User-friendly function to reinterpret text.
        """
        self.eval() # Set the model to evaluation mode
        with torch.no_grad(): # Disable gradient calculation
            # 1. Encode the input text using the inherited X-encoder
            inputs = self.tokenizer(text, return_tensors='pt').to(self.encoder.device)
            latent_vector = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask']).last_hidden_state
            
            # 2. Use Y's own decoder to generate new text from the latent vector
            # The .generate() method is ideal for creative text generation
            outputs = self.decoder.generate(
                inputs_embeds=latent_vector,
                max_new_tokens=max_new_tokens,
                do_sample=True, # Use sampling for more creative and less repetitive results
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Decode the generated token IDs into text
            new_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return new_text

# --- 2. Conceptual Training Block for Y ---
def train_artist_y_conceptual():
    """
    This is a conceptual guide for training Y's decoder.
    It requires a new, creative dataset of (latent_vector, new_text) pairs.
    """
    print("\n--- Conceptual Training Process for ArtistY ---")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. First, load the fully trained ArtistX model
    print("1. Loading the trained ArtistX model...")
    model_X = ArtistX()
    # Ensure you have the 'artist_x_best_model.pth' file from training X
    model_X.load_state_dict(torch.load("artist_x_best_model.pth"))
    model_X.to(DEVICE)
    model_X.eval()
    print("✅ ArtistX loaded.")

    # 2. Create an instance of ArtistY and load the encoder from X
    print("2. Preparing ArtistY...")
    model_Y = ArtistY().to(DEVICE)
    model_Y.load_encoder_from_x(model_X)
    
    # 3. CRITICAL: Freeze the encoder's weights. We only want to train the decoder.
    for param in model_Y.encoder.parameters():
        param.requires_grad = False
    print("✅ Y's encoder is frozen. Only the decoder will be trained.")

    # 4. Prepare the optimizer for only the decoder's parameters
    optimizer = torch.optim.AdamW(model_Y.decoder.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    # 5. Prepare a new "creative" dataset. This is a crucial, non-trivial step.
    #    You need a dataset where the input is a sentence and the target is a *different*,
    #    artistically related sentence.
    #    creative_dataloader = ... 

    # 6. Conceptual Training Loop
    # model_Y.train()
    # for epoch in range(num_epochs):
    #     for batch in creative_dataloader:
    #         original_text = batch['original_text']
    #         creative_target_text = batch['creative_text']
    #         
    #         # Get latent vector from the original text via the frozen encoder
    #         # Train the decoder to produce the creative_target_text from that vector
    #         ...
    
    print("✅ Conceptual training setup for Y is complete.")
    return model_Y


# --- 3. Main Execution Block ---
if __name__ == '__main__':
    print("🎭 'Reinterpreting Artist' ArtistY setup and demonstration.")
    
    # This function prepares a Y model with a trained encoder for demonstration
    Y = train_artist_y_conceptual()
    
    # Since Y's decoder is not actually trained, the output will be random.
    # This demonstrates how the 'reinterpret' function works.
    
    original_text = "a painter who remembers what he drew"
    
    new_text = Y.reinterpret(original_text)

    print("\n--- Y's Reinterpretation Test ---")
    print(f"Original Text: {original_text}")
    print(f"Y's Reinterpreted Text: {new_text}")
    print("\n(Note: Since Y's decoder is untrained, the output is currently random.)")