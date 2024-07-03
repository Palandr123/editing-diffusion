import random
import logging
from functools import partial

import streamlit as st
import torch
import torchvision.transforms.functional as TF
import diffusers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

from editing_diffusion.diffusion_models import SDXLEditingPipeline
from editing_diffusion.editing import CustomAttentionProcessor
from editing_diffusion.llm_objects import spot_objects


_SG_RES = 64
SEED = 1
save_aux = True
random.seed(SEED)
torch.manual_seed(SEED)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
DEVICE = "cuda:1"
LLM_DEVICE = "cuda:2"
MODEL_NAME = "google/gemma-7b-it"
MODEL_PARAMS = {
    "max_new_tokens": 200,
}


@st.cache_resource
def load_stable_diffusion():
    global DEVICE
    base = SDXLEditingPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        use_safetensors=True,
        torch_dtype=torch.float16,
        variant="fp16",
        use_onnx=False,
    )
    base.to(DEVICE)
    base.scheduler = diffusers.DDPMScheduler.from_config(base.scheduler.config)
    return base


@st.cache_resource
def load_llm():
    global MODEL_NAME
    global LLM_DEVICE
    llm = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval().to(LLM_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer, llm


def resave_aux_key(module, *args, old_key="attn", new_key="last_attn"):
    module._aux[new_key] = module._aux[old_key]


def resize(x):
    return TF.resize(x, _SG_RES, antialias=True)


def stash_to_aux(
    module,
    args,
    kwargs,
    output,
    mode,
    key="last_feats",
    args_idx=None,
    kwargs_key=None,
    fn_to_run=None,
):
    to_save = None
    if mode == "args":
        to_save = input
        if args_idx is not None:
            to_save = args[args_idx]
    elif mode == "kwargs":
        assert kwargs_key is not None
        to_save = kwargs[kwargs_key]
    elif mode == "output":
        to_save = output
    if fn_to_run is not None:
        to_save = fn_to_run(to_save)
    try:
        global save_aux
        if not save_aux:
            len_ = len(module._aux[key])
            del module._aux[key]
            module._aux[key] = [None] * len_ + [to_save]
        else:
            module._aux[key][-1] = module._aux[key][-1].cpu()
            module._aux[key].append(to_save)
    except:
        try:
            del module._aux[key]
        except:
            pass
        module._aux = {key: [to_save]}


def store_results(original_image):
    # Replace with your preferred storage mechanism (e.g., list, database)
    results = []
    if st.session_state.get("past_results"):
        results = st.session_state["past_results"]
    results.append({
        "original_image": original_image,
        # "manipulated_object": manipulated_object,
        # "manipulation": manipulation,
        # "manipulated_image": manipulated_image,
    })
    st.session_state["past_results"] = results[-5:] 


base = load_stable_diffusion()
attn_greenlist = []
for i in range(0, len(base.unet.up_blocks) - 2):
    for j in range(len(base.unet.up_blocks[i].attentions)):
        base_name = f"up_blocks.{i}.attentions.{j}.transformer_blocks"
        for name, module in (
            base.unet.up_blocks[i].attentions[j].transformer_blocks.named_children()
        ):
            for name_child, _ in module.named_children():
                if name_child == "attn2":
                    attn_greenlist.append(base_name + f".{name}.{name_child}")

tokenizer, llm = load_llm()


def generate_image(prompt: str) -> Image:
    global save_aux
    global base
    global attn_greenlist
    global llm
    global tokenizer
    save_aux = True
    for name, block in base.unet.named_modules():
        if isinstance(
            block,
            (
                diffusers.models.unet_2d_blocks.CrossAttnDownBlock2D,
                diffusers.models.unet_2d_blocks.CrossAttnUpBlock2D,
                diffusers.models.unet_2d_blocks.UNetMidBlock2DCrossAttn,
            ),
        ):
            for attn_name, attn in block.named_modules():
                full_name = name + "." + attn_name
                if "attn2" not in attn_name or (
                    attn_greenlist and full_name not in attn_greenlist
                ):
                    continue
                attn.processor = CustomAttentionProcessor(_SG_RES, save_aux)
    handle1 = base.unet.up_blocks[2].register_forward_hook(
        partial(stash_to_aux, mode="output"), with_kwargs=True
    )
    handle2 = (
        base.unet.up_blocks[0]
        .attentions[1]
        .transformer_blocks[3]
        .attn2.register_forward_hook(resave_aux_key)
    )
    logger.info(f"Generating image #{i}")
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    out = base(
        prompt=[prompt],
        num_inference_steps=300,
        generator=generator,
        save_aux=save_aux,
        latents=None,
    )
    

    aux = base.get_sg_aux()
    aux_idx = 0
    processed_aux = {
        k: torch.utils._pytree.tree_map(
            lambda x: x[aux_idx : aux_idx + 1].repeat_interleave(1, 0).cpu(), v
        )
        for k, v in aux.items()
    }
    handle1.remove()
    handle2.remove()
    return out.images[0], processed_aux
    


prompt = st.text_input("Enter a prompt for the image generation:")
if st.button("Generate Image"):
    if prompt:
        generated_image, processed_aux = generate_image(prompt)
        st.image(generated_image)
        objects = spot_objects(tokenizer, llm, prompt, LLM_DEVICE, **MODEL_PARAMS)
        st.header("Objects to be detected:")

        # Loop through dictionary items
        for key, value in objects["objects"]:
            # Display key
            st.write(f"**{key}**")

            for item in value:
                if item:
                    st.write(f"- {item}")

        
        store_results(generated_image)
        st.subheader("Last Manipulation Results")
        if st.session_state.get("past_results"):
            for result in st.session_state["past_results"]:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(result["original_image"])
                    st.caption("Original Image")
                # with col2:
                #     st.image(result["manipulated_image"])
                #     st.caption(f"Manipulated {result['manipulated_object']}")
                #     st.write(f"Manipulation: ({result
    else:
        st.warning("Please enter a prompt!")