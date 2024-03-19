import ast

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
import torch


def get_key_objects(
    tokenizer: PreTrainedTokenizerFast,
    model: PreTrainedModel,
    message: str,
    device: str | torch.device,
    **model_params
) -> tuple[dict[str, str], str]:
    input_ids = tokenizer.encode(message, return_tensors="pt")
    generated_ids = model.generate(
        input_ids=input_ids.to(device),
        **model_params,
    )
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    response = response[len(message) :]

    # Extracting key objects
    key_objects_part = response.split("Objects:")[1]
    start_index = key_objects_part.index("[")
    end_index = key_objects_part.rindex("]") + 1
    objects_str = key_objects_part[start_index:end_index]

    # Converting string to list
    parsed_objects = ast.literal_eval(objects_str)

    # Extracting additional negative prompt
    bg_prompt = response.split("Background:")[1].split("\n")[0].strip()
    negative_prompt = response.split("Negation:")[1].strip()

    parsed_result = {
        "objects": parsed_objects,
        "bg_prompt": bg_prompt,
        "neg_prompt": negative_prompt,
    }
    return parsed_result, response
