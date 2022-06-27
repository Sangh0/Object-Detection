import copy

import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    """
    params:
        - embed_dim: The length of embedding sequence
        - n_head: The number of multi head
        - feedforward_dim: The channel number of last layer of backbone
        - dropout: The rate of dropout
    """
    def __init__(
        self,
        embed_dim=512,
        n_head=8,
        feedforward_dim=2048,
        dropout=0.1,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, n_head, dropout=dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(embed_dim)
        
        self.linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, src, pos=None):
        q = k = self.pos_add(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.attn_drop(src2)
        src = self.attn_norm(src)
        
        src2 = self.linear2(self.dropout1(self.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.ffn_norm(src)
        return src
    
    def pos_add(self, src, pos):
        return src if pos == None else src + pos
    
class TransformerEncoder(nn.Module):
    """
    params:
        - encoder_layer: Pre-defined encoder layer class
        - layer_num: Repeat count of encoder layers
        - norm: apply normalization layer or None
    """
    def __init__(self, encoder_layer, layer_num, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = clone_layer(encoder_layer, layer_num)
        self.norm = norm
        
    def forward(self, src, pos=None):
        out = src
        for layer in self.layers:
            out = layer(out, pos)
        if self.norm:
            out = self.norm(out)
        return out
    
class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        n_head=8,
        feedforward_dim=2048,
        dropout=0.1,
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, n_head, dropout=dropout)        
        self.mh_attn = nn.MultiheadAttention(embed_dim, n_head, dropout=dropout)
        
        self.linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, pos=None, query_pos=None):
        q = k = self.pos_add(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        tgt2 = self.mh_attn(self.pos_add(tgt, query_pos), self.pos_add(memory, pos), value=memory)[0]
        tgt = tgt + self.dropout2(tgt)
        tgt = self.norm2(tgt)
        
        tgt2 = self.linear2(self.dropout(self.relu(self.linear1(tgt))))
        
        tgt = tgt + self.dropout3(tgt2)
        return tgt
    
    def pos_add(self, tensor, pos):
        return tensor if pos == None else tensor + pos

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, layer_num, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = clone_layer(decoder_layer, layer_num)
        self.norm = norm
        
    def forward(self, tgt, memory, pos=None, query_pos=None):
        out = tgt
        for layer in self.layers:
            out = layer(out, memory, pos, query_pos)
        if self.norm:
            out = self.norm(out)
        return out
    
class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        n_head=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        feedforward_dim=2048,
        dropout=0.1,
    ):
        super(Transformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(embed_dim, n_head, feedforward_dim, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.embed_dim = embed_dim
        
        decoder_layer = TransformerDecoderLayer(embed_dim, n_head, feedforward_dim, dropout)
        decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
    def forward(self, src, query_embed, pos_embed):
        B, C, H, W = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)
        
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, pos=pos_embed)
        hs = self.decoder(tgt, memory, query_pos=query_embed)
        return hs.permute(1, 0, 2), memory.permute(1, 2, 0).view(B, C, H, W)
    
def clone_layer(layer, layer_num):
    return [copy.deepcopy(layer) for _ in range(layer_num)]