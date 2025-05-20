import time
import sys
sys.path.append('./')
from PIL import Image
import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List

import torch
import os
import gc
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')

def init():
    unet = UNet2DConditionModel.from_pretrained(
        base_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    unet.requires_grad_(False)
    tokenizer_one = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

    text_encoder_one = CLIPTextModel.from_pretrained(
        base_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_path,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
        )
    vae = AutoencoderKL.from_pretrained(base_path,
                                        subfolder="vae",
                                        torch_dtype=torch.float16,
    )

    # "stabilityai/stable-diffusion-xl-base-1.0",
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        base_path,
        subfolder="unet_encoder",
        torch_dtype=torch.float16,
    )

    UNet_Encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    tensor_transfrom = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
        )

    pipe = TryonPipeline.from_pretrained(
            base_path,
            unet=unet,
            vae=vae,
            feature_extractor= CLIPImageProcessor(),
            text_encoder = text_encoder_one,
            text_encoder_2 = text_encoder_two,
            tokenizer = tokenizer_one,
            tokenizer_2 = tokenizer_two,
            scheduler = noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
    )
    pipe.unet_encoder = UNet_Encoder
    return pipe, tensor_transfrom


pipe, tensor_transfrom = init()
pipe.enable_sequential_cpu_offload()


def show_device():
    if pipe.unet is not None:
        d_unet = next(pipe.unet.parameters()).device
        print(f'UNet is on: {d_unet}')

    if pipe.text_encoder is not None:
        d_text_encoder = next(pipe.text_encoder.parameters()).device
        print(f'Text Encoder 1 is on: {d_text_encoder}')

    if pipe.text_encoder_2 is not None:
        d_text_encoder_2 = next(pipe.text_encoder_2.parameters()).device
        print(f'Text Encoder 2 is on: {d_text_encoder_2}')

    if pipe.image_encoder is not None:
        d_image_encoder = next(pipe.image_encoder.parameters()).device
        print(f'Image Encoder is on: {d_image_encoder}')

    if pipe.vae is not None:
        d_vae = next(pipe.vae.parameters()).device
        print(f'VAE is on: {d_vae}')

    # for obj in gc.get_objects():
    #     if isinstance(obj, torch.Tensor) and obj.is_cuda:
    #         print(f"Deleting Tensor: {obj} - Size: {obj.size()}")


def move2cpu():
    if pipe.unet is not None:
        pipe.unet.to('cpu')

    if pipe.text_encoder is not None:
        pipe.text_encoder.to('cpu')

    if pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.to('cpu')

    if pipe.image_encoder is not None:
        pipe.image_encoder.to('cpu')

    if pipe.vae is not None:
        pipe.vae.to('cpu')
    torch.cuda.empty_cache()


def move_model(is_move2cpu, model):
    if model == 'unet':
        if pipe.unet is not None:
            pipe.unet.to('cpu' if is_move2cpu else 'cuda:0')

    elif model == 'text_encoder':
        if pipe.text_encoder is not None:
            pipe.text_encoder.to('cpu' if is_move2cpu else 'cuda:0')

    elif model == 'text_encoder_2':
        if pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.to('cpu' if is_move2cpu else 'cuda:0')

    elif model == 'image_encoder':
        if pipe.image_encoder is not None:
            pipe.image_encoder.to('cpu' if is_move2cpu else 'cuda:0')

    elif model == 'vae':
        if pipe.vae is not None:
            pipe.vae.to('cpu' if is_move2cpu else 'cuda:0')
    torch.cuda.empty_cache()

def delete_all_tensor_on_gpu():
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            del obj
    torch.cuda.empty_cache()


def start_tryon(dict_, garm_img_, garment_des, denoise_steps_, seed_):
    start_time = time.time()

    garm_img_ = garm_img_.convert("RGB").resize((720,1280))
    human_img_orig = dict_["background"].convert("RGB")
    
    human_img = human_img_orig.resize((720,1280))
    mask = pil_to_binary_mask(dict_['layers'][0].convert("RGB").resize((720, 1280)))
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)
    mask_time = time.time()
    print(f"preprocess-mask: {mask_time - start_time:.2f} s")

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    # verbosity = getattr(args, "verbosity", None)
    pose_img = args.func(args,human_img_arg)
    pose_img = pose_img[:,:,::-1]
    pose_img = Image.fromarray(pose_img).resize((720, 1280))
    pose_time = time.time()
    print(f"preprocess-pose: {pose_time - mask_time:.2f} s")
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )

                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )
                    encode_prompt_time = time.time()
                    print(f"preprocess-encode_prompt: {encode_prompt_time - pose_time:.2f} s")

                    pose_img = tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                    garm_tensor = tensor_transfrom(garm_img_).unsqueeze(0).to(device,torch.float16)
                    generator = torch.Generator(device).manual_seed(seed_) if seed_ is not None else None
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device,torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                        num_inference_steps=denoise_steps_,
                        generator=generator,
                        strength = 1.0,
                        pose_img = pose_img.to(device,torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                        cloth = garm_tensor.to(device,torch.float16),
                        mask_image=mask,
                        image=human_img, 
                        height=1280,
                        width=720,
                        ip_adapter_image = garm_img_.resize((720,1280)),
                        guidance_scale=2.0,
                    )[0]

    print(f"total: {time.time() - start_time:.2f} s")

    return images[0], mask_gray
    # return images[0], mask_gray


garm_list = os.listdir(os.path.join(example_path,"cloth"))
garm_list_path = [os.path.join(example_path,"cloth",garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path,"human"))
human_list_path = [os.path.join(example_path,"human",human) for human in human_list]

human_ex_list = []
for ex_human in human_list_path:
    ex_dict= {}
    ex_dict['background'] = ex_human
    ex_dict['layers'] = None
    ex_dict['composite'] = None
    human_ex_list.append(ex_dict)

##default human


image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.Markdown("## IDM-VTON 👕👔👚")
    gr.Markdown("Virtual Try-on with your image and garment image. Check out the [source codes](https://github.com/yisol/IDM-VTON) and the [model](https://huggingface.co/yisol/IDM-VTON)")
    with gr.Row():
        with gr.Column():
            imgs = gr.ImageEditor(sources='upload', type="pil", label='Human. Mask with pen or use auto-masking', interactive=True)

            example = gr.Examples(
                inputs=imgs,
                examples_per_page=10,
                examples=human_ex_list
            )

        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="pil")
            with gr.Row(elem_id="prompt-container"):
                with gr.Row():
                    prompt = gr.Textbox(placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts", show_label=False, elem_id="prompt")
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=8,
                examples=garm_list_path)
        with gr.Column():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            masked_img = gr.Image(label="Masked image output", elem_id="masked-img",show_share_button=False)
        with gr.Column():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            image_out = gr.Image(label="Output", elem_id="output-img",show_share_button=False)




    with gr.Column():
        try_button = gr.Button(value="Try-on")
        with gr.Accordion(label="Advanced Settings", open=False):
            with gr.Row():
                denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=40, value=20, step=1)
                seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)

    with gr.Column():
        show_button = gr.Button(value="Show Device")
    with gr.Row():
        switch_u = gr.Checkbox(label="unet on cpu", value=True)
        switch_te = gr.Checkbox(label="text_encoder on cpu", value=True)
        switch_te2 = gr.Checkbox(label="text_encoder_2 on cpu", value=True)
        switch_ie = gr.Checkbox(label="image_encoder on cpu", value=True)
        switch_vae = gr.Checkbox(label="vae on cpu", value=True)

    try_button.click(fn=start_tryon, inputs=[imgs, garm_img, prompt, denoise_steps, seed], outputs=[image_out,masked_img], api_name='tryon')
    show_button.click(fn=show_device)
    switch_u.change(fn=move_model, inputs=[switch_u, gr.Textbox(value='unet', visible=False)], outputs=None)
    switch_te.change(fn=move_model, inputs=[switch_te, gr.Textbox(value='text_encoder', visible=False)], outputs=None)
    switch_te2.change(fn=move_model, inputs=[switch_te2, gr.Textbox(value='text_encoder_2', visible=False)],outputs=None)
    switch_ie.change(fn=move_model, inputs=[switch_ie, gr.Textbox(value='image_encoder', visible=False)], outputs=None)
    switch_vae.change(fn=move_model, inputs=[switch_vae, gr.Textbox(value='vae', visible=False)], outputs=None)

image_blocks.launch()

