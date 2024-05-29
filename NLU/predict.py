import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from NLU.utils import MODEL_CLASSES, get_intent_labels, get_slot_labels, init_logger, load_tokenizer


logger = logging.getLogger(__name__)


def get_args(file_path):
    # f"https://huggingface.com/{model_dir}/blob/main/training_args.bin"
    return torch.load(file_path)
# torch.hub.load_state_dict_from_url(f"https://huggingface.com/{model_dir}/blob/main/training_args.bin")
# torch.load(os.path.join(model_dir, "training_args.bin"))
# torch.hub.load_state_dict_from_url(f"https://huggingface.co/{model_dir}/blob/main/training_args.bin")


def convert_input_file_to_tensor_dataset(
    line,
    args,
    tokenizer,
    pad_token_label_id,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    #Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    tokens = []
    slot_label_mask = []
    for word in line:
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            word_tokens = [unk_token]  # For handling the bad-encoded word
        tokens.extend(word_tokens)
        # Use the real label id for the first token of the word, and padding ids for the remaining tokens
        slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

    # Account for [CLS] and [SEP]
    special_tokens_count = 2
    if len(tokens) > args.max_seq_len - special_tokens_count:
        tokens = tokens[: (args.max_seq_len - special_tokens_count)]
        slot_label_mask = slot_label_mask[: (args.max_seq_len - special_tokens_count)]

    # Add [SEP] token
    tokens += [sep_token]
    token_type_ids = [sequence_a_segment_id] * len(tokens)
    slot_label_mask += [pad_token_label_id]

    # Add [CLS] token
    tokens = [cls_token] + tokens
    token_type_ids = [cls_token_segment_id] + token_type_ids
    slot_label_mask = [pad_token_label_id] + slot_label_mask

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = args.max_seq_len - len(input_ids)
    input_ids = input_ids + ([pad_token_id] * padding_length)
    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
    slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

    all_input_ids.append(input_ids)
    all_attention_mask.append(attention_mask)
    all_token_type_ids.append(token_type_ids)
    all_slot_label_mask.append(slot_label_mask)
    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)
    return all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask


def predict_nlu(string, file_path, model_dir, slot_path, intent_path):
    # load model and args
    string = string.strip().split()
    args = get_args(file_path)
    # print("Hello: ",args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = MODEL_CLASSES[args.model_type][1].from_pretrained(
            model_dir, args=args, intent_label_lst = get_intent_labels(intent_path), slot_label_lst=get_slot_labels(slot_path)
        )
        model.to(device)
        model.eval()
    except Exception:
        raise Exception("Some model files might be missing...")

    # logger.info(args)

    intent_label_lst = get_intent_labels(intent_path)
    slot_label_lst = get_slot_labels(slot_path)

    # Convert input file to TensorDataset
    pad_token_label_id = args.ignore_index
    tokenizer = load_tokenizer(args)
    all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask = convert_input_file_to_tensor_dataset(string, args, tokenizer, pad_token_label_id)

    # all_slot_label_mask = None
    intent_preds = None
    slot_preds = None

# =======================================================================
    with torch.no_grad():
        inputs = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "intent_label_ids": None,
            "slot_labels_ids": None,
        }
        if args.model_type != "distilbert":
            inputs["token_type_ids"] = all_token_type_ids
        outputs = model(**inputs)
        _, (intent_logits, slot_logits) = outputs[:2]

        # Intent Prediction
        intent_preds = intent_logits.detach().cpu().numpy()
        # Slot prediction
        if args.use_crf:
            # decode() in `torchcrf` returns list with best index directly
            slot_preds = np.array(model.crf.decode(slot_logits))
        else:
            slot_preds = slot_logits.detach().cpu().numpy()
        all_slot_label_mask = all_slot_label_mask.detach().cpu().numpy()

# =======================================================================
    intent_preds = np.argmax(intent_preds)

    if not args.use_crf:
        slot_preds = np.argmax(slot_preds, axis=1)

    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

    for i in range(slot_preds.shape[0]):
        for j in range(slot_preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

    line = ""
    
    for word, pred in zip(string, slot_preds_list[0]):
        if pred == "O":
            line = line + word + " "
        else:
            line = line + "[{}:{}] ".format(word, pred)
    print("<{}> -> {}\n".format(intent_label_lst[intent_preds], line.strip()))

    return "<{}> -> {}\n".format(intent_label_lst[intent_preds], line.strip())


# slot_path = "E:/Data_SLU_journal/STREAMLIT_SLU/streamlit_app/NLU/slot_label.txt"
# intent_path = "E:/Data_SLU_journal/STREAMLIT_SLU/streamlit_app/NLU/intent_label.txt"
# model_dir = "E:/Data_SLU_journal/NLU_MODEL_V100/NLU_model/JointIDSF/checkpoint_NLU_slotfilling_250/JointIDSF_PhoBERTencoder/4e-5/0.15/100"
# string = "trời sáng quá, đóng rèm lại đi"
# predict(string, model_dir, slot_path, intent_path)