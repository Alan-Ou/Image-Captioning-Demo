import torch

from transformers import BertTokenizer
from PIL import Image
import argparse

from models import caption
from datasets import coco, utils
from configuration import Config
import os


def model_predict(img_path):
    image_path = img_path
    version = 'v3'
    config = Config()

    local_path = os.path.dirname(__file__)
    # local_path = r'C:\Users\sc18co\.cache\torch\hub\project'
    model = torch.hub.load(local_path, version, source='local', pretrained=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

    image = Image.open(image_path)
    image = coco.val_transform(image)
    image = image.unsqueeze(0)

    max_length = config.max_position_embeddings
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    caption, cap_mask = caption_template, mask_template

    with torch.no_grad():
        model.eval()
        for i in range(config.max_position_embeddings - 1):
            predictions = model(image, caption, cap_mask)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)

            if predicted_id[0] == 102:
                output = caption
                break

            caption[:, i + 1] = predicted_id[0]
            cap_mask[:, i + 1] = False

        output = caption

    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    # result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result.capitalize()
