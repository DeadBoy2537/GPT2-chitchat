import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from train_origin import create_model
import torch.nn.functional as F
import copy

from flask import Flask, request, jsonify

app = Flask(__name__)

PAD = '[PAD]'
pad_id = 0


def set_interact_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
                        help='模型参数')
    parser.add_argument('--log_path', default='data/interacting_mmi.log', type=str, required=False,
                        help='interact_mmi日志存放位置')
    parser.add_argument('--voca_path', default='vocab/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--dialogue_model_path', default='dialogue_model/', type=str, required=False,
                        help='dialogue_model路径')
    parser.add_argument('--mmi_model_path', default='mmi_model/', type=str, required=False,
                        help='互信息mmi_model路径')
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False, help="保存聊天记录的文件路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_len', type=int, default=25, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=5, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    parser.add_argument('--batch_size', type=int, default=5, help='批量生成response，然后经过MMI模型进行筛选')
    parser.add_argument('--debug', action='store_true', help='指定该参数，可以查看生成的所有候选的reponse，及其loss')
    return parser.parse_args()


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocab size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 2
    top_k = min(top_k, logits[0].size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        for logit in logits:
            indices_to_remove = logit < torch.topk(logit, top_k)[0][..., -1, None]
            logit[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for index, logit in enumerate(logits):
            indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
            logit[indices_to_remove] = filter_value
    return logits


def main():
    args = set_interact_args()
    logger = create_logger(args)
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    tokenizer = BertTokenizer(vocab_file=args.voca_path)
    # 对话model
    dialogue_model = GPT2LMHeadModel.from_pretrained(args.dialogue_model_path)
    dialogue_model.to(device)
    dialogue_model.eval()
    # 互信息mmi model
    mmi_model = GPT2LMHeadModel.from_pretrained(args.mmi_model_path)
    mmi_model.to(device)
    mmi_model.eval()
    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/mmi_samples.txt', 'a', encoding='utf8')
        samples_file.write("聊天记录{}:\n".format(datetime.now()))
        # 存储聊天记录，每个utterance以token的id的形式进行存储
    history = []
    print('开始和chatbot聊天')
    app.run(host='127.0.0.1', port=5000, debug=True)
    # while True:
    # text = input('user')
    # print(main(text))

            # mmi模型的输入
            if args.debug:
                print("candidate response:")
            samples_file.write("candidate response:\n")
            min_loss = float('Inf')
            best_response = ""
            for response in candidate_responses:
                mmi_input_id = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头
                mmi_input_id.extend(response)
                mmi_input_id.append(tokenizer.sep_token_id)
                for history_utr in reversed(history[-args.max_history_len:]):
                    mmi_input_id.extend(history_utr)
                    mmi_input_id.append(tokenizer.sep_token_id)
                mmi_input_tensor = torch.tensor(mmi_input_id).long().to(device)
                out = mmi_model(input_ids=mmi_input_tensor, labels=mmi_input_tensor)
                loss = out[0].item()
                if args.debug:
                    text = tokenizer.convert_ids_to_tokens(response)
                    print("{} loss:{}".format("".join(text), loss))
                samples_file.write("{} loss:{}\n".format("".join(text), loss))
                if loss < min_loss:
                    best_response = response
                    min_loss = loss
            history.append(best_response)
            text = tokenizer.convert_ids_to_tokens(best_response)
            print("chatbot:" + "".join(text))
            if args.save_samples_path:
                samples_file.write("chatbot:{}\n".format("".join(text)))
        except KeyboardInterrupt:
            if args.save_samples_path:
                samples_file.close()
            break


if __name__ == '__main__':
    main()
