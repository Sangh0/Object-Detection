import torch
import torch.nn as nn

class PositionEncoding(nn.Module):
    def __init__(self, n_dim=512):
        super(PositionEncoding, self).__init__()
        self.embed_dim = n_dim // 2

        self.row_embed = nn.Embedding(50, self.embed_dim)
        self.col_embed = nn.Embedding(50, self.embed_dim)
        self._reset_parameters_()
        
    def forward(self, x):
        H, W = x.size()[-2:]
        i = torch.arange(W, device=x.device)
        j = torch.arange(H, device=x.device)
        x_embed = self.col_embed(i)
        y_embed = self.row_embed(j)
        
        """ 
        shape of x_embed: (7,256)
        unsqueeze with dim=0 -> (1,7,256)
        repeat with (H,1,1) -> (7,7,256)
        
        shape of y_embed: (7,256)
        unsqueeze with dim=0 -> (7,1,256)
        repeat with (H,1,1) -> (7,7,256)
        """
        pos = torch.cat([
            x_embed.unsqueeze(0).repeat(H, 1, 1),
            y_embed.unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).permute(2,0,1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos
    
    def _reset_parameters_(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)