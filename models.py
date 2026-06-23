import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class ImageBranch(nn.Module):
    """
    Extracts features from building visual chips using EfficientNet-B0.
    """
    def __init__(self, pretrained=False, embedding_dim=128):
        super().__init__()
        if pretrained:
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            # Freeze the convolutional feature extractor so that FL clients only
            # fine-tune the lightweight classification head.  Training all ~4M
            # EfficientNet parameters on tiny per-client shards (often < 100 samples)
            # causes catastrophic overfitting; the global FedAvg of such overfitted
            # models averages out to noise and the global model never improves.
            # The last conv block (features[-1]) is left trainable so the backbone
            # can still adapt its high-level representations to aerial disaster imagery.
            for name, param in self.backbone.named_parameters():
                # Freeze all blocks except the final conv block ("features.8")
                if not name.startswith("features.8"):
                    param.requires_grad = False
        else:
            self.backbone = efficientnet_b0(weights=None)
        
        # Output features from EfficientNet-B0 conv head is 1280
        in_features = self.backbone.classifier[1].in_features
        # Bypass default classifier
        self.backbone.classifier = nn.Identity()
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.fc(features)
        return embedding

class StructuredBranch(nn.Module):
    """
    Extracts features from geographic coordinates and USGS seismic parameters.
    """
    def __init__(self, input_dim=9, embedding_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class MultiModalFusionModel(nn.Module):
    """
    Fuses the image embedding (128-dim) and structured embedding (64-dim)
    and classifies the building damage state into 4 categories.
    """
    def __init__(self, num_classes=4, pretrained=False):
        super().__init__()
        self.image_branch = ImageBranch(pretrained=pretrained, embedding_dim=128)
        self.structured_branch = StructuredBranch(input_dim=9, embedding_dim=64)
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, struct_features):
        img_embed = self.image_branch(img)
        struct_embed = self.structured_branch(struct_features)
        
        # Concatenate features along the channel/feature dimension
        fused = torch.cat([img_embed, struct_embed], dim=1)
        logits = self.fusion_fc(fused)
        return logits

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss to address extreme imbalance in damage classes.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Weights for each class
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            # Shift alpha to targets device
            alpha_t = self.alpha.to(targets.device)[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss