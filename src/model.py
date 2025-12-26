import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

class ViTDet(nn.Module):
    def __init__(self, num_classes, num_queries=100, img_size=640, pretrained_path=None, backbone_name='vit_tiny_patch16_224'):
        super().__init__()
        
        # 1. ViT Backbone (Encoder) using timm
        # pretrained=False because we load manually
        # img_size=img_size allows timm to adjust pos_embed automatically if we were downloading,
        # but since we load manually, we rely on load_checkpoint to resize.
        self.backbone = timm.create_model(backbone_name, pretrained=False, img_size=img_size)
        self.hidden_dim = self.backbone.embed_dim
        
        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained(pretrained_path)
        
        # 2. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=3)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # 3. Object Queries
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)
        
        # 4. Prediction Heads
        self.class_embed = nn.Linear(self.hidden_dim, num_classes + 1)
        self.bbox_embed = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 4),
            nn.Sigmoid()
        )
        
    def load_pretrained(self, path):
        print(f"Loading pretrained weights from {path}")
        if path.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
                checkpoint = load_file(path)
            except ImportError:
                print("Error: 'safetensors' library not found. Please install it using: pip install safetensors")
                raise
        else:
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            except TypeError:
                # For older PyTorch versions
                checkpoint = torch.load(path, map_location='cpu')
        
        # If checkpoint is a dict with 'model' or 'state_dict' key, extract it
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            elif 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
                
        # Resize pos_embed if needed
        if 'pos_embed' in checkpoint:
            pos_embed_checkpoint = checkpoint['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            
            # Get model's expected number of patches
            # timm models usually have patch_embed.num_patches
            num_patches = self.backbone.patch_embed.num_patches
            # Check for prefix tokens (CLS token, etc.)
            num_extra_tokens = getattr(self.backbone, 'num_prefix_tokens', 1)
            
            if pos_embed_checkpoint.shape[1] != num_patches + num_extra_tokens:
                print(f"Resizing pos_embed from {pos_embed_checkpoint.shape} to match model ({num_patches + num_extra_tokens} tokens).")
                
                orig_size = int(math.sqrt(pos_embed_checkpoint.shape[1] - num_extra_tokens))
                new_size = int(math.sqrt(num_patches))
                
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint['pos_embed'] = new_pos_embed

        # Load into backbone
        msg = self.backbone.load_state_dict(checkpoint, strict=False)
        print(f"Loaded weights with msg: {msg}")

    def forward(self, x):
        # x: [Batch, 3, H, W]
        
        # timm forward_features returns [B, N, D] (including CLS token)
        features = self.backbone.forward_features(x) 
        
        memory = features.permute(1, 0, 2) 
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, x.shape[0], 1)
        
        hs = self.decoder(query_embed, memory)
        hs = hs.permute(1, 0, 2)
        
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs)
        
        return {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord
        }
