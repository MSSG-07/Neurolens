"""Two-path model; each batch row must be a same-patient 2D/3D pair."""
from torch import nn
from .encoder2d import Encoder2D
from .encoder3d import Encoder3D
from .alignment import ProjectionHead


class NeuroLens(nn.Module):
    def __init__(self, in_channels=4, base_channels=16, projection_dim=256, classes=4, classification_classes=2):
        super().__init__()
        self.encoder2d = Encoder2D(in_channels, base_channels*2, projection_dim, classification_classes)
        self.encoder3d = Encoder3D(in_channels, base_channels, projection_dim, classes)
        self.project2d = ProjectionHead(projection_dim, projection_dim)
        self.project3d = ProjectionHead(projection_dim, projection_dim)

    def forward(self, image2d, image3d):
        out2, out3 = self.encoder2d(image2d), self.encoder3d(image3d)
        return {"classification_logits": out2["logits"], "embedding2d": self.project2d(out2["embedding"]),
                "embedding3d": self.project3d(out3["embedding"]), "segmentation_logits": out3["segmentation_logits"],
                "features2d": out2["features"], "pooled2d": out2["pooled"], "pooled3d": out3["pooled"]}
