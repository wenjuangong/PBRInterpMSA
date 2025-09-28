import pickle
import torch
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer
import pickle
import re


def traindata():
    data_iemo = 'D:\PyCharm Community Edition 2023.2.3\pythonProject\IEMOCAP_features.pkl'
    videoIDs, videoSpeakers, videoLabels, videoText, \
        videoAudio, videoVisual, videoSentence, trainVid, \
        testVid = pickle.load(open(data_iemo, 'rb'), encoding='latin1')

    list_ = []
    train = []
    for i in trainVid:
        for j in videoSentence[i]:
            cleaned_text = re.sub(r'[^\w\s]', '', j)
            tokens = cleaned_text.split()
            train.append([tokens])
    for i in trainVid:
        for j in videoVisual[i]:
            list_.append(j)
    result = [x + [y] for x, y in zip(train, list_)]
    list_.clear()
    for i in trainVid:
        for j in videoAudio[i]:
            list_.append(j)
    result = [x + [y] for x, y in zip(result, list_)]
    import csv
    # 打开CSV文件
    with open('D:\PyCharm Community Edition 2023.2.3\pythonProject\CENet - jo\CENet-main\dataset\pos.csv', 'r',
              encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        # 将每一行转换为一个列表
        next(csv_reader)
        pos = [[element for element in row if element] for row in csv_reader]
    converted_list = [[float(item) for item in sublist] for sublist in pos]
    result = [x + [y] for x, y in zip(result, converted_list)]
    with open('D:\PyCharm Community Edition 2023.2.3\pythonProject\CENet - jo\CENet-main\dataset\senti.csv', 'r',
              encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        # 将每一行转换为一个列表
        next(csv_reader)
        senti = [[element for element in row if element] for row in csv_reader]
    converted_list2 = [[float(item) for item in sublist] for sublist in senti]
    result = [x + [y] for x, y in zip(result, converted_list2)]
    list_.clear()
    for i in trainVid:
        for j in videoVisual[i]:
            list_.append(j)
    result = [x + [y] for x, y in zip(result, list_)]
    list_.clear()
    for i in trainVid:
        for j in videoAudio[i]:
            list_.append(j)
    result = [x + [y] for x, y in zip(result, list_)]
    list_.clear()
    for i in trainVid:
        for j in videoLabels[i]:
            list_.append(j)
    result = [x + [y] for x, y in zip(result, list_)]
    list_.clear()
    for i in trainVid:
        for j in videoIDs[i]:
            list_.append(j)
    result = [x + [y] for x, y in zip(result, list_)]
    return result


def testdata():
    data_iemo = 'D:\PyCharm Community Edition 2023.2.3\pythonProject\IEMOCAP_features.pkl'
    videoIDs, videoSpeakers, videoLabels, videoText, \
        videoAudio, videoVisual, videoSentence, trainVid, \
        testVid = pickle.load(open(data_iemo, 'rb'), encoding='latin1')

    list_ = []
    train = []
    for i in testVid:
        for j in videoSentence[i]:
            cleaned_text = re.sub(r'[^\w\s]', '', j)
            tokens = cleaned_text.split()
            train.append([tokens])
    for i in testVid:
        for j in videoVisual[i]:
            list_.append(j)
    result = [x + [y] for x, y in zip(train, list_)]
    list_.clear()
    for i in testVid:
        for j in videoAudio[i]:
            list_.append(j)
    result = [x + [y] for x, y in zip(result, list_)]
    import csv
    # 打开CSV文件
    with open('D:\PyCharm Community Edition 2023.2.3\pythonProject\CENet - jo\CENet-main\dataset\pos_test.csv', 'r',
              encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        # 将每一行转换为一个列表
        next(csv_reader)
        pos = [[element for element in row if element] for row in csv_reader]
    converted_list = [[float(item) for item in sublist] for sublist in pos]
    result = [x + [y] for x, y in zip(result, converted_list)]
    with open('D:\PyCharm Community Edition 2023.2.3\pythonProject\CENet - jo\CENet-main\dataset\senti_test.csv', 'r',
              encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        # 将每一行转换为一个列表
        next(csv_reader)
        senti = [[element for element in row if element] for row in csv_reader]
    converted_list2 = [[float(item) for item in sublist] for sublist in senti]
    result = [x + [y] for x, y in zip(result, converted_list2)]
    list_.clear()
    for i in testVid:
        for j in videoVisual[i]:
            list_.append(j)
    result = [x + [y] for x, y in zip(result, list_)]
    list_.clear()
    for i in testVid:
        for j in videoAudio[i]:
            list_.append(j)
    result = [x + [y] for x, y in zip(result, list_)]
    list_.clear()
    for i in testVid:
        for j in videoLabels[i]:
            list_.append(j)
    result = [x + [y] for x, y in zip(result, list_)]
    list_.clear()
    for i in testVid:
        for j in videoIDs[i]:
            list_.append(j)
    result = [x + [y] for x, y in zip(result, list_)]
    return result


class InputFeatures(object):

    def __init__(self, input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic,
                 input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual_ids = visual_ids
        self.acoustic_ids = acoustic_ids
        self.pos_ids = pos_ids
        self.senti_ids = senti_ids
        self.polarity_ids = polarity_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob


def convert_to_features(args, examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):

        words, visual, acoustic, pos_ids, senti_ids, visual_ids, acoustic_ids, label_id, segment = example

        tokens, inversions, = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)  # 分词后词的列表 [a lot of sad parts]
            inversions.extend([idx] * len(tokenized))  # [0 1 2 3 4]

        # Check inversion
        assert len(tokens) == len(inversions)

        aligned_pos_ids = []
        aligned_senti_ids = []

        for inv_idx in inversions:
            aligned_pos_ids.append(pos_ids[inv_idx])
            aligned_senti_ids.append(senti_ids[inv_idx])

        # visual = np.array(aligned_visual)
        visual = np.array(visual)
        visual_ids = np.array(visual_ids)
        acoustic = np.array(acoustic)
        acoustic_ids = np.array(acoustic_ids)
        pos_ids = aligned_pos_ids
        senti_ids = aligned_senti_ids

        # Truncate input if necessary 截断

        if len(tokens) > max_seq_length - 3:
            tokens = tokens[: max_seq_length - 3]
            words = words[: max_seq_length - 3]
            pos_ids = pos_ids[: max_seq_length - 3]
            senti_ids = senti_ids[: max_seq_length - 3]

        input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids = prepare_sentilare_input(
            args, tokens, visual_ids, acoustic_ids, pos_ids, senti_ids, visual, acoustic, tokenizer
        )
        # Check input length
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                visual_ids=visual_ids,
                acoustic_ids=acoustic_ids,
                pos_ids=pos_ids,
                senti_ids=senti_ids,
                polarity_ids=polarity_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features


def prepare_sentilare_input(args, tokens, visual_ids, acoustic_ids, pos_ids, senti_ids, visual, acoustic,
                            tokenizer):  # 生成BERT需要的输入形式
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP] + [SEP]
    pos_ids = [4] + pos_ids + [4] + [4]
    senti_ids = [2] + senti_ids + [2] + [2]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)  # 生成长度为inputids，元素全为0的列表
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)
    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    pos_ids += [4] * pad_length
    senti_ids += [2] * pad_length
    polarity_ids = [5] * len(input_ids)
    input_mask += padding
    segment_ids += padding

    return input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids


def get_tokenizer(args):
    return RobertaTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False)


def get_appropriate_dataset(args, data):
    tokenizer = get_tokenizer(args)

    features = convert_to_features(args, data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_visual_ids = torch.tensor(
        [f.visual_ids for f in features], dtype=torch.long)
    all_acoustic_ids = torch.tensor(
        [f.acoustic_ids for f in features], dtype=torch.long)
    all_pos_ids = torch.tensor(
        [f.pos_ids for f in features], dtype=torch.long)
    all_senti_ids = torch.tensor(
        [f.senti_ids for f in features], dtype=torch.long)
    all_polarity_ids = torch.tensor(
        [f.polarity_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor([f.acoustic for f in features], dtype=torch.float)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual_ids,
        all_acoustic_ids,
        all_pos_ids,
        all_senti_ids,
        all_polarity_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )
    return dataset


def set_up_data_loader(args):
    # with open(args.data_path, "rb") as handle:
    #     data = pickle.load(handle)

    # train_data = data["train"]
    # dev_data = data["dev"]
    # test_data = data["test"]
    train_data = traindata()
    test_data = testdata()
    train_dataset = get_appropriate_dataset(args, train_data)
    # dev_dataset = get_appropriate_dataset(args, dev_data)
    test_dataset = get_appropriate_dataset(args, test_data)

    num_train_optimization_steps = (
            int(
                len(train_dataset) / args.train_batch_size /
                args.gradient_accumulation_step
            )
            * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True
    )

    # dev_dataloader = DataLoader(
    #     dev_dataset, batch_size=args.dev_batch_size, shuffle=True, drop_last=True
    # )
    #
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    return (
        train_dataloader,
        # dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


# import argparse
#
#
# def parser_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", type=str,
#                         choices=["mosi", "mosei"], default="mosei")
#     parser.add_argument("--data_path", type=str, default='./dataset/MOSEI_16_sentilare_unaligned_data.pkl')
#     parser.add_argument("--max_seq_length", type=int, default=50)
#     parser.add_argument("--n_epochs", type=int, default=20)
#     parser.add_argument("--train_batch_size", type=int, default=64)
#     parser.add_argument("--beta_shift", type=float, default=1.0)
#     parser.add_argument("--dropout_prob", type=float, default=0.5)
#     parser.add_argument(
#         "--model",
#         type=str,
#         choices=["bert-base-uncased", "xlnet-base-cased", "roberta-base"],
#         default="roberta-base")
#     parser.add_argument("--model_name_or_path",
#                         default='D:/PyCharm Community Edition 2023.2.3/pythonProject/CENet - data/CENet-main/pretrained_model/sentilare_model/',
#                         type=str,
#                         help="Path to pre-trained model or shortcut name")
#     parser.add_argument("--learning_rate", type=float, default=6e-5)
#     parser.add_argument("--weight_decay", type=float, default=0)
#     parser.add_argument("--gradient_accumulation_step", type=int, default=1)
#     parser.add_argument("--test_step", type=int, default=20)
#     parser.add_argument("--max_grad_norm", type=int, default=2)
#     parser.add_argument("--warmup_proportion", type=float, default=0.4)
#
#     parser.add_argument("--adam_epsilon", default=1e-8, type=float,
#                         help="Epsilon for Adam optimizer.")
#     return parser.parse_args()
#
#
# args = parser_args()
# (train_data_loader, test_data_loader, num_train_optimization_steps,) = set_up_data_loader(args)