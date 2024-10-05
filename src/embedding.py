import pandas as pd
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class CustomSentenceTransformer(nn.Module):
    def __init__(self, base_model_name, output_dim):
        super(CustomSentenceTransformer, self).__init__()
        self.base_model = SentenceTransformer(base_model_name)
        self.linear = nn.Linear(self.base_model.get_sentence_embedding_dimension(), output_dim)

    def forward(self, input_texts):
        # Encode text to embeddings
        embeddings = self.base_model.encode(input_texts, convert_to_tensor=True)
        # Transform embeddings to the desired size
        return self.linear(embeddings)

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def decode_strings(df):
    """Decode byte strings to regular strings and handle NaNs."""
    for col in df.columns:
        if col not in ['Date', 'Label']:
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x.strip('b"').strip("'"))
            df[col] = df[col].astype(str)  # Ensure all values are strings
    return df

def preprocess(text):
    """Preprocess and tokenize the text."""
    return text.lower()

def tokenize_columns(df):
    """Tokenize text columns and create a new columns dictionary."""
    new_columns = {}
    for col in df.columns:
        if col not in ['Date', 'Label']:
            new_columns[col + "_tokens"] = df[col].apply(preprocess)
    return new_columns

def create_embeddings(df, model):
    """Create sentence embeddings for the text columns."""
    for col in df.columns:
        if col not in ['Date', 'Label']:
            df[col + "_embedding"] = df[col].apply(lambda x: model([x])[0].tolist())  # Process list of texts
    return df

def create_embedding_dataframe(df):
    """Create a DataFrame with embedding columns and the Date column."""
    embedding_columns = [col for col in df.columns if "_embedding" in col]
    embedding_df = pd.concat([df[['Date']], df[embedding_columns]], axis=1)
    return embedding_df

def save_embeddings(embedding_df, output_path):
    """Save the embeddings to a CSV file."""
    embedding_df.to_csv(output_path, index=False)

def embedding_main(data_file_path, output_file_path):
    df = load_data(data_file_path)
    df = decode_strings(df)
    
    # Initialize the custom model with 32-dimensional embeddings
    model = CustomSentenceTransformer('paraphrase-MiniLM-L6-v2', output_dim=32)

    df = create_embeddings(df, model)
    embedding_df = create_embedding_dataframe(df)

    save_embeddings(embedding_df, output_file_path)
    print(f"\nEmbeddings have been saved to '{output_file_path}'.")
