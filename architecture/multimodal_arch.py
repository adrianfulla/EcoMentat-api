import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.resnet50_fpn import CAM_PRED

ATMOS_KEYS = ["NO2", "CO", "PM2.5", "PM10"]

class VisualProbModule(nn.Module):
    """Envuelve CAM_PRED y devuelve logits 1D (con microbatch y flip opcional)."""
    def __init__(self, backbone_ctor, state_dict_path: str, visual_microbatch: int = 4, use_flip: bool = False):
        super().__init__()
        self.backbone = backbone_ctor()
        # Carga del checkpoint de la rama visual base (igual que en entrenamiento)
        sd = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(sd, strict=True)
        self.visual_microbatch = max(1, int(visual_microbatch))
        self.use_flip = use_flip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        outs = []
        for s in range(0, B, self.visual_microbatch):
            e = min(B, s + self.visual_microbatch)
            xb = x[s:e]
            if self.use_flip:
                # Infiere original y flip, pero conserva sólo el original (como en tu script)
                xb_flip = torch.flip(xb, dims=[3])
                x_pair = torch.cat([xb, xb_flip], dim=0)
                cam, logits = self.backbone(x_pair, )   # logits [2m,1]
                v = logits.view(-1)[0::2]             # sólo originales
                del cam, logits, x_pair, xb_flip
            else:
                cam, logits = self.backbone(xb)       # logits [m,1]
                v = logits.view(-1)
                del cam, logits
            outs.append(v)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(outs, dim=0)  # [B]

class AtmosBranch(nn.Module):
    def __init__(self, in_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(inplace=True), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(inplace=True), nn.BatchNorm1d(32), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class FusionHead(nn.Module):
    def __init__(self, in_dim=17):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(inplace=True), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.mlp(x).view(-1)

class MultiModalModel(nn.Module):
    """Salida final: logits. La prob. visual (sigmoid) es el feature 1-D para la fusión."""
    def __init__(self, backbone_ctor, state_dict_path: str, visual_microbatch: int):
        super().__init__()
        self.visual = VisualProbModule(backbone_ctor, state_dict_path, visual_microbatch, use_flip=False)
        self.atmos  = AtmosBranch(in_dim=len(ATMOS_KEYS))   # = 4
        self.fusion = FusionHead(in_dim=1 + 16)             # 1 (prob visual) + 16 (ramo atmos)

    def freeze_visual(self, freeze=True):
        for p in self.visual.parameters():
            p.requires_grad = not freeze

    def forward(self, images, atmos_feats):
        v_logits = self.visual(images)                 # [B]
        v_prob   = torch.sigmoid(v_logits).unsqueeze(1)  # [B,1]
        a_feats  = self.atmos(atmos_feats)             # [B,16]
        fused    = torch.cat([v_prob, a_feats], dim=1) # [B,17]
        logits   = self.fusion(fused)                  # [B]
        return logits


def build_model(visual_ckpt_path: str, visual_microbatch: int = 2):
    def backbone_ctor():
        # num_classes=1 (binario), pretrained=False
        return CAM_PRED(num_classes=1, pretrained=False)
    return MultiModalModel(backbone_ctor, visual_ckpt_path, visual_microbatch)
