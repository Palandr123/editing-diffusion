import numpy as np
from numpy.typing import NDArray
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL.Image import Image
import inflect

from editing_diffusion.detectors.base import Detector

p = inflect.engine()


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
    
    def detect(self, image: Image, mode: str, device: torch.device | str, score_threshold: float, nms_threshold: float) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if mode == "attribute":
            target_objects = [x for x in self.attribute_count]
        elif mode == "primitive":
            target_objects = [x for x in self.primitive_count]
        if len(target_objects) == 0:
            return []
        
        texts = [[f"image of {p.a(obj)}" for obj in target_objects]]
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        width, height = image.size
        target_sizes = torch.Tensor([[height, width]])
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0
        )

        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=width)
        boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=height)
        boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=width)
        boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=height)
        boxes = boxes.cpu().detach()
        boxes = np.array(
            [
                [x_min / width, y_min / height, x_max / width, y_max / height]
                for (x_min, y_min, x_max, y_max), score in zip(boxes, scores)
                if score >= score_threshold
            ]
        )
        scores = np.array(
            [
                score.cpu().detach().numpy()
                for score in scores
                if score >= score_threshold
            ]
        )
        return scores, boxes
