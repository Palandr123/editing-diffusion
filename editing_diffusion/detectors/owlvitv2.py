import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

from editing_diffusion.detectors.base import Detector


class OWLViTv2Detector(Detector):
    def __init__(
        self,
        device: str | torch.device,
        attr_detection_threshold: float = 0.6,
        prim_detection_threshold: float = 0.2,
        nms_threshold: float = 0.5,
    ):
        super().__init__()
        self.default_attr_detection_threshold = attr_detection_threshold
        self.default_prim_detection_threshold = prim_detection_threshold
        self.default_nms_threshold = nms_threshold

        self.processor = Owlv2Processor.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        )
        owl_vit_model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        )
        self.model = owl_vit_model.eval().to(device)
