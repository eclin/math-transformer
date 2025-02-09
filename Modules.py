import torch
import torch.nn as nn

SEED = 69
torch.manual_seed(SEED)

if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available. Using CPU.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Tokenizer, represented as an embedding layer
EMB_SIZE = 32

class Tokenizer(nn.Module):
    def __init__(self, num_tokens: int, max_length: int, emb_size: int):
        super().__init__()
        self.max_length = max_length
        self.token_embedding = torch.nn.Embedding(num_tokens, emb_size)
        self.positional_embedding = torch.nn.Embedding(max_length, emb_size)

    def forward(self, token_ids: torch.tensor) -> torch.tensor:
        # token_ids is a LongTensor of token IDs with shape [batch_size, max_length]

        batch_size, seq_len = token_ids.shape
        # shape: [batch_size, seq_len]
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(device)

        return self.token_embedding(token_ids) + self.positional_embedding(positions)


class Attention(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.Q = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.K = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.V = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X: torch.tensor, Z: torch.tensor, mask: torch.tensor = None) -> torch.tensor:
        query = self.Q(X)
        key = self.K(Z)
        value = self.V(Z)
        score = query @ key.transpose(-2, -1)

        score = score / (self.emb_dim ** 0.5)
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))
        output = self.softmax(score) @ value

        return output

class MHAttention(nn.Module):
    def __init__(self, emb_dim: int, heads: int = 4):
        super().__init__()
        assert emb_dim % heads == 0
        self.head_emb_dim = emb_dim // heads
        self.heads = heads
        self.emb_dim = emb_dim
        self.attention_heads = nn.ModuleList([Attention(self.head_emb_dim) for _ in range(heads)])
        self.output_layer = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim)

    def forward(self, X: torch.tensor, Z: torch.tensor, mask: bool = True) -> torch.tensor:
        batch_size, x_seq_len, _ = X.shape
        _, z_seq_len, _ = Z.shape
        causal_mask = None
        if mask:
            causal_mask = torch.tril(torch.ones(x_seq_len, z_seq_len)).bool().to(device)
        Y = torch.concat([attention(
            X.view(batch_size, x_seq_len, self.heads, self.head_emb_dim)[:, :, idx, :],
            Z.view(batch_size, z_seq_len, self.heads, self.head_emb_dim)[:, :, idx, :],
            mask=causal_mask) for idx, attention in enumerate(self.attention_heads)], dim=-1)
        output = self.output_layer(Y)

        return output

class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.offset = torch.nn.Parameter(torch.zeros(emb_dim))
        self.epsilon = 1e-5

    def forward(self, X: torch.tensor) -> torch.tensor:
        """
        Normalizes layer activations
        X should have shape [batch_size, x_seq_len, emb_dim]
        """
        # m = (X.sum(dim=-1) / self.emb_dim).unsqueeze(-1)  # [batch_size, x_seq_len, 1]
        m = X.mean(dim=-1, keepdim=True)
        v = ((X - m) ** 2).mean(dim=-1, keepdim=True) # / self.emb_dim).unsqueeze(-1) + self.epsilon  # [batch_size, x_seq_len, 1]
        output = (X - m) / torch.sqrt(v + self.epsilon)  # [batch_size, x_seq_len, emb_dim]
        output = output * self.scale + self.offset

        return output


class EDTransformer(nn.Module):
    """
    An encoder-decoder transformer
    """

    def __init__(
        self,
        emb_dim: int,
        num_tokens: int,
        max_context_length: int,
        max_primary_length: int,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        heads: int = 4
    ):
        super().__init__()
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.heads = heads
        self.emb_dim = emb_dim
        self.num_tokens = num_tokens
        self.max_context_length = max_context_length
        self.max_primary_length = max_primary_length

        self.tokenizer = Tokenizer(max(max_context_length, max_primary_length), emb_dim)

        # Encoder
        self.encoder_layer_mhattentions = nn.ModuleList([MHAttention(emb_dim, heads=heads) for _ in range(encoder_layers)])
        self.encoder_first_layer_norms = nn.ModuleList([LayerNorm(emb_dim) for _ in range(encoder_layers)])
        self.encoder_second_layer_norms = nn.ModuleList([LayerNorm(emb_dim) for _ in range(encoder_layers)])
        self.encoder_fc_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=emb_dim)
        ) for _ in range(encoder_layers)])

        # Decoder
        self.decoder_layer_first_mhattentions = nn.ModuleList([MHAttention(emb_dim, heads=heads) for _ in range(decoder_layers)])
        self.decoder_layer_second_mhattentions = nn.ModuleList([MHAttention(emb_dim, heads=heads) for _ in range(decoder_layers)])
        self.decoder_first_layer_norms = nn.ModuleList([LayerNorm(emb_dim) for _ in range(decoder_layers)])
        self.decoder_second_layer_norms = nn.ModuleList([LayerNorm(emb_dim) for _ in range(decoder_layers)])
        self.decoder_third_layer_norms = nn.ModuleList([LayerNorm(emb_dim) for _ in range(decoder_layers)])
        self.decoder_fc_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=emb_dim)
        ) for _ in range(decoder_layers)])

        self.unembedding = nn.Linear(in_features=emb_dim, out_features=num_tokens, bias=False)

    def forward(self, Z: torch.tensor, X: torch.tensor) -> torch.tensor:
        # Z has shape [batch_size, max_tokens] representing the context sequence (equations)
        # X has shape [batch_size, max_tokens] representing the primary sequence (answers)

        Z = self.tokenizer(Z)
        for l in range(self.encoder_layers):
            Z = Z + self.encoder_layer_mhattentions[l](Z, Z, mask=False)
            Z = self.encoder_first_layer_norms[l](Z)
            Z = Z + self.encoder_fc_layers[l](Z)
            Z = self.encoder_second_layer_norms[l](Z)

        X = self.tokenizer(X)
        for l in range(self.decoder_layers):
            X = X + self.decoder_layer_first_mhattentions[l](X, X, mask=True)
            X = self.decoder_first_layer_norms[l](X)
            X = X + self.decoder_layer_second_mhattentions[l](X, Z, mask=False)
            X = self.decoder_second_layer_norms[l](X)
            X = X + self.decoder_fc_layers[l](X)
            X = self.decoder_third_layer_norms[l](X)

        output = self.unembedding(X)

        return output
