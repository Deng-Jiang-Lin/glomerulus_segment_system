import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image


class SegModel:
    def __init__(
        self,
        name: str,
        in_chns: int = 3,
        cls: int = 1,
        weight_path: str = r"D:\glomerulus_segment_system\pretrained\UNetpp_best_model.pth"
    ):
        self.name = name
        self.in_chns = in_chns
        self.cls = cls
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._build_model(weight_path)
        self.model.eval()

        self.input_size = (512, 512)
        self.transform = self._build_transform()

    @property
    def parameters_number(self) -> float:
        return round(
            sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6,
            3
        )

    # ---------------------
    # Model
    # ---------------------
    def _build_model(self, weight_path: str):
        model = smp.UnetPlusPlus(
            in_channels=self.in_chns,
            classes=self.cls,
            encoder_weights=None
        ).to(self.device)

        state_dict = torch.load(weight_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)

        return model

    # ---------------------
    # Transform
    # ---------------------
    def _build_transform(self):
        return A.Compose([
            A.Resize(*self.input_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # ---------------------
    # Predict
    # ---------------------
    def predict(self, image_path: str) -> np.ndarray:
        image = np.array(Image.open(image_path).convert("RGB"))

        transformed = self.transform(image=image)["image"]
        tensor = (
            torch.from_numpy(transformed)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        with torch.no_grad():
            pred = self.model(tensor)
            if self.cls == 1:
                mask = (torch.sigmoid(pred) > 0.5)
                mask = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
            else:
                mask = torch.argmax(pred, dim=1)
                mask = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
        return mask
