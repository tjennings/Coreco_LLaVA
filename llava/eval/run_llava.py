import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO


import argparse
import glob
import os
import json5
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Any, Tuple, Union, Dict
from collections import OrderedDict


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    image_processor, image_token_len, mm_use_im_start_end, model, model_name, tokenizer = setup_models(args)

    conv_mode = set_conv_mode(args, model_name)

    qs = args.query
    outputs = query(args, image_processor, image_token_len, mm_use_im_start_end, model, qs, tokenizer)

    print(outputs)


def setup_models(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "mpt" in model_name.lower():
        model = LlavaMPTForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                    use_cache=True).cuda()
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                      use_cache=True).cuda()

    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    vision_tower = model.get_model().vision_tower[0]
    if vision_tower.device.type == 'meta':
        vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16,
                                                       low_cpu_mem_usage=True).cuda()
        model.get_model().vision_tower[0] = vision_tower
    else:
        vision_tower.to(device='cuda', dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    return image_processor, image_token_len, mm_use_im_start_end, model, model_name, tokenizer


def set_conv_mode(args, model_name):
    conv_mode = ""
    if "v1" in model_name.lower():
        conv_mode = "llava_v1"

    elif "mpt" in model_name.lower():
        conv_mode = "mpt_multimodal"
    else:
        conv_mode = "multimodal"


    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            '[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode,
                                                                                                              args.conv_mode,
                                                                                                              args.conv_mode))
    else:
        args.conv_mode = conv_mode

    return conv_mode


def query(args, image, image_processor, image_token_len, mm_use_im_start_end, model, qs, tokenizer):
    if mm_use_im_start_end:
        qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
    else:
        qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=256,
            stopping_criteria=[stopping_criteria])
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


class Node:
    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def get_template(self) -> str:
        return self.get("template", "{}")

    def get_question(self) -> str:
        return self.data["q"]

    def get_branches(self) -> Dict[str, Any]:
        return self.data.get("branches", {})

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

class Workflow:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes

    @staticmethod
    def from_json(json_data: List[Dict[str, Any]]):
        return Workflow([Node(node_data) for node_data in json_data])


class Inquisitor:
    def __init__(self, workflow: Workflow, model: Any, args: argparse.Namespace):
        self.workflow = workflow
        self.model = model
        self.args = args
        self.prefix = args.prefix or ""
        self.suffix = args.suffix or ""

    def ask(self, image: torch.Tensor) -> str:
        context: List[Tuple[str, str]] = []
        return f"{self.prefix} {self._traverse_workflow(image, context, self.workflow.nodes)} {self.suffix}"

    def _traverse_workflow(self, image: torch.Tensor, context: List[Tuple[str, str]], nodes: List[Node]) -> str:
        responses = []
        for node in nodes:
            response = self._process_node(image, context, node)
            if response:
                responses.append(response.strip())
        join_str = node.get("join", ' ')
        return join_str.join(responses)

    def _process_node(self, image: torch.Tensor, context: List[Tuple[str, str]], node: Node) -> str:
        answer, context = self._query(image, context, node)

        branches = node.get_branches()
        if answer in branches:
            branch = branches[answer]
            if isinstance(branch, list):
                return self._traverse_workflow(image, context, [Node(b) for b in branch])
            else:
                return self._process_node(image, context, Node(branch))
        elif not answer:
            return ""
        else:
            return node.get_template().format(answer)

    def _query(self, image: torch.Tensor, context: List[Tuple[str, str]], node: Node) -> Tuple[str, List[Tuple[str, str]]]:
        template = "Question: {}, Answer: {}."
        question = node.get_question()
        prompt = " ".join([template.format(q, a) for q, a in context]) + f" Question: {question} Answer:"
        answer = self.model.generate({"image": image, "prompt": prompt})[0].lower()
        answer = self._deduplicate(answer)

        #print(f"{prompt} {answer}")

        #if 'not enough information' in answer:
        #    raise Exception("Not enough information response")

        return answer, context + [(question, answer)]

    @staticmethod
    def _deduplicate(answer: str) -> str:
        return " ".join(OrderedDict.fromkeys(answer.split(" ")))

class ImageDataset(Dataset):
    def caption_path(self, path):
        return os.path.join(os.path.dirname(path), f"{os.path.basename(path).split('.')[0]}.txt")

    def caption_exists(self, path):
        caption_path = self.caption_path(path)
        return os.path.exists(caption_path)

    def __init__(self, dir, args):
        self.dir = dir
        self.args = args

        image_paths = glob.glob(os.path.join(args.path, '**/*.*'), recursive=True)

        if args.overwrite:
            self.paths = [p for p in image_paths if
                          p.endswith(('jpg', 'jpeg', 'png', 'webp'))]
        else:
            self.paths = [p for p in image_paths if p.endswith(('jpg', 'jpeg', 'png', 'webp'))
                          and not self.caption_exists(p)]

    def __len__(self):
        return (len(self.paths))

    def __getitem__(self, idx):
        path = self.paths[idx]

        if self.args.resize is not None:
            self.resize_and_save(path, self.args.resize)

        raw_image = self.resize_image(Image.open(path).convert("RGB"), 768)
        return {"image": raw_image, "caption_path": self.caption_path(path)}

    @staticmethod
    def resize_and_save(path, resize):
        img = Image.open(path)
        ImageDataset.resize_image(img, resize).save(path)

    @staticmethod
    def resize_image(image, target_size):
        # Get the current width and height of the image
        width, height = image.size

        # Check which dimension (width or height) is the maximum
        max_dimension = max(width, height)

        # If the image is smaller than target, abort
        if max(width, height) <= target_size:
            return image

        # Calculate the scale factor to resize the image
        scale_factor = 1
        if max_dimension > target_size:
            scale_factor = target_size / max_dimension

        # Calculate the new width and height using the scale factor
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize the image while maintaining the aspect ratio
        resized_image = image.resize((new_width, new_height))

        # Return the resized image
        return resized_image

    @staticmethod
    def collate_fn(batch):
        return batch


def load_workflow(file_path: str) -> Workflow:
    with open(file_path, 'r') as f:
        return Workflow.from_json(json5.load(f))


def main(args):
    args.conv_mode = None
    print("Starting up ...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Loading models ...")
    image_processor, image_token_len, mm_use_im_start_end, model, model_name, tokenizer = setup_models(args)

    dataset = ImageDataset(args.path, args)
    loader = DataLoader(dataset, batch_size=1, num_workers=5, collate_fn=ImageDataset.collate_fn)


    set_conv_mode(args, model_name)


    print("Captioning  ...")
    for batch in tqdm(loader):
        image, caption_path = batch[0]["image"], batch[0]["caption_path"]

        try:
            qs = "provide a detailed description of this image."
            answer = query(args, image, image_processor, image_token_len, mm_use_im_start_end, model, qs, tokenizer)
            answer = answer.lower().replace("the image", "", 1)
            with open(caption_path, "w") as f:
                f.write(answer.split("\n\n")[0])
        except Exception as e:
            print(f"Failed to process {caption_path}, {str(e)}")
            raise e


if __name__ == "__main__":
    args = argparse.ArgumentParser("described")
    args.add_argument("--path", type=str, required=True, help="Path to images to be captioned")
    args.add_argument("--overwrite", default=False, action="store_true", help="Overwrite existing captions")
    args.add_argument("--model_name", type=str, default="/mnt/van_gogh/ckpt/LLaVA-13B-v0/", help="a string applied at the beginning of each caption")
    args.add_argument("--prefix", type=str, help="a string applied at the beginning of each caption")
    args.add_argument("--suffix", type=str, help="a string applied at the end of each caption")
    args.add_argument("--resize", type=int, help="additionally, resize and save the image where the longest side is the provided maximum ")
    args = args.parse_args()

    main(args)



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
#     parser.add_argument("--image-file", type=str, required=True)
#     parser.add_argument("--query", type=str, required=True)
#     parser.add_argument("--conv-mode", type=str, default=None)
#     args = parser.parse_args()
#
#     eval_model(args)