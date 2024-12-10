import torch
import torch.nn as nn
import numpy as np

# Paramètres du dataset
NUM_SAMPLES = 1000      # nombre d'exemples dans le dataset
SEQ_LEN = 1900          # longueur de chaque série temporelle
BATCH_SIZE = 32
EPOCHS = 2000
LATENT_DIM = 16         # dimension du bruit pour le générateur

# Paramètres du transformeur
EMBED_DIM = 64
NHEAD = 4
NUM_LAYERS = 2
FFN_HIDDEN_DIM = 128


class Generator(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, latent_dim=LATENT_DIM, emb_dim=EMBED_DIM, nhead=NHEAD, num_layers=NUM_LAYERS, ffn_hidden_dim=FFN_HIDDEN_DIM):
        super(Generator, self).__init__()
        
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.embedding = nn.Linear(latent_dim, emb_dim)  # on transforme le bruit en embedding
        transformer_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=ffn_hidden_dim)
        self.transformer = nn.TransformerDecoder(transformer_layer, num_layers=num_layers)
        
        self.positional_encoding = self._build_positional_encoding(emb_dim, seq_len)
        self.output_linear = nn.Linear(emb_dim, 1)  # on sort 1 valeur par timestep
        self.register_buffer('tgt_mask', generate_square_subsequent_mask(seq_len))
        
    def _build_positional_encoding(self, d_model, max_len):
        # Positional encoding (classique)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[0::, 0, 0::2] = torch.sin(position * div_term)
        pe[0::, 0, 1::2] = torch.cos(position * div_term)
        return pe  # shape (max_len, 1, d_model)

    def forward(self, z):
        # z: (batch, latent_dim)
        batch_size = z.size(0)
        
        # On va répéter l'embedding sur la longueur de séquence, 
        # ou on peut simplement fournir un "début" vide et laisser le transformer "décoder".
        # Ici, on va considérer qu'on envoie un bruit répété sur seq_len steps.
        # Une autre approche : utiliser un embedding de shape (seq_len, batch, emb_dim)
        
        # Par simplicité, on va initialiser un token de départ qui sera le même pour tous,
        # puis le transformer decoder va générer la séquence.
        
        # Initialiser une entrée "cible" vide (juste du bruit broadcasté)
        # On part de vecteurs nuls comme target embeddings.
        tgt = torch.zeros(self.seq_len, batch_size, EMBED_DIM, device=z.device)  
        
        # Le state latent va être une clé/mémoire dans le decoder:
        # On va utiliser l'embedding du bruit comme "memory" (encodé)
        memory = self.embedding(z)  # (batch, emb_dim)
        # On reshape memory pour qu'il fasse (seq_len, batch, emb_dim) 
        # Ici on ne dispose pas vraiment d'une séquence en memory, 
        # on peut juste répéter ce vecteur sur un seul pas de temps ou plusieurs.
        # On va juste répéter le memory sur un pas de temps:
        memory = memory.unsqueeze(0)  # (1, batch, emb_dim)
        
        # Ajout du positional encoding sur la target
        tgt = tgt + self.positional_encoding.to(tgt.device)
        
        out = self.transformer(tgt, memory, tgt_mask=self.tgt_mask)
        # out: (seq_len, batch, emb_dim)
        
        # On projette sur une valeur par pas de temps
        out = self.output_linear(out)  # (seq_len, batch, 1)
        out = out.permute(1, 0, 2).squeeze(-1)  # (batch, seq_len)
        return out


class Discriminator(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, emb_dim=EMBED_DIM, nhead=NHEAD, num_layers=NUM_LAYERS, ffn_hidden_dim=FFN_HIDDEN_DIM):
        super(Discriminator, self).__init__()
        
        self.seq_len = seq_len
        # Un transformer encoder pour classifier le signal comme réel ou faux
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=ffn_hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.input_linear = nn.Linear(1, emb_dim)
        self.positional_encoding = self._build_positional_encoding(emb_dim, seq_len)
        self.output_linear = nn.Linear(emb_dim, 1)
        
    def _build_positional_encoding(self, d_model, max_len):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[0::, 0, 0::2] = torch.sin(position * div_term)
        pe[0::, 0, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        # x: (batch, seq_len)
        batch_size = x.size(0)
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        x = self.input_linear(x)  # (batch, seq_len, emb_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch, emb_dim)
        
        x = x + self.positional_encoding.to(x.device)
        out = self.transformer(x)  # (seq_len, batch, emb_dim)
        
        # On pool en prenant le token moyen ou le dernier
        # Ici on va simplement prendre le premier token comme "classement"
        out = out[0, :, :]  # (batch, emb_dim)
        out = self.output_linear(out)  # (batch, 1)
        return out

# Masque causal pour le générateur si on utilise un transformer decoder
def generate_square_subsequent_mask(sz: int):
    mask = torch.triu(torch.ones(sz, sz), 1)  # upper-triangular matrix of ones
    mask = mask.masked_fill(mask==1, float('-inf'))
    return mask