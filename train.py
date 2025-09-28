from __future__ import absolute_import, division, print_function

import argparse
import random
from pytorch_transformers.modeling_roberta import RobertaConfig
import torch
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import time
from torch.nn import L1Loss, MSELoss
from pytorch_transformers import WarmupLinearSchedule, AdamW
from networks.SentiLARE import RobertaForSequenceClassification
from networks.subnet import global_configs
from utils.databuilder import set_up_data_loader
from utils.set_seed import set_random_seed, seed
from utils.metric import score_model
from utils.hard_exmples import hard_fea
from networks.subnet.global_configs import *
import os

os.environ["WANDB_DISABLED"] = "true"
os.environ['WANDB_MODE'] = 'dryrun'


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        choices=["mosi", "mosei"], default="mosei")
    parser.add_argument("--data_path", type=str, default='./dataset/MOSEI_16_sentilare_unaligned_data.pkl')
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=global_configs.batch_size)
    parser.add_argument("--dev_batch_size", type=int, default=global_configs.batch_size)
    parser.add_argument("--test_batch_size", type=int, default=global_configs.batch_size)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--beta_shift", type=float, default=1.0)
    parser.add_argument("--dropout_prob", type=float, default=0.5)
    parser.add_argument(
        "--model",
        type=str,
        choices=["bert-base-uncased", "xlnet-base-cased", "roberta-base"],
        default="roberta-base")
    parser.add_argument("--model_name_or_path", default='D:/PyCharm Community Edition 2023.2.3/pythonProject/CENet - data/CENet-main/pretrained_model/sentilare_model/', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--gradient_accumulation_step", type=int, default=1)
    parser.add_argument("--test_step", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=int, default=2)
    parser.add_argument("--warmup_proportion", type=float, default=0.4)
    parser.add_argument("--seed", type=seed, default=6758, help="integer or 'random'")  # 6758
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    return parser.parse_args()

def add_gaussian_noise(tensor, mean=0, std=1):
    noise = torch.randn(tensor.size()).to(DEVICE) * std + mean
    return tensor + noise

def add_uniform_noise(tensor, low=-0.1, high=0.1):
    noise = torch.rand(tensor.size()).to(DEVICE) * (high - low) + low
    return tensor + noise

def add_random_noise(tensor):
    noise_type = random.choice(['gaussian', 'uniform'])
    if noise_type == 'gaussian':
        return add_gaussian_noise(tensor, mean=0, std=0.1)
    elif noise_type == 'uniform':
        return add_uniform_noise(tensor, low=-0.1, high=0.1)

def prep_for_training(args, num_train_optimization_steps: int):
    config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=1, finetuning_task='sst')
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config, pos_tag_embedding=True, senti_embedding=True, polarity_embedding=True)
    model.to(DEVICE)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    CE_params = ['CE']
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if
                not any(nd in n for nd in no_decay) and not any(nd in n for nd in CE_params)
            ],
            "weight_decay": args.weight_decay,
        },
        {"params": model.roberta.encoder.CE.parameters(), 'lr': args.learning_rate, "weight_decay": args.weight_decay},
        {
            "params": [
                p for n, p in param_optimizer if
                any(nd in n for nd in no_decay) and not any(nd in n for nd in CE_params)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        t_total=num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def train_epoch(args, model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    preds = []
    labels = []
    tr_loss = 0
    nb_tr_steps = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        label_ids = torch.where(label_ids == 0, torch.tensor(1.0), torch.tensor(-1.0))
        # for i in range(num_gauss_noise):
        #     input_ids[i] = add_random_noise(input_ids[i])
        #     visual_ids[i] = add_random_noise(visual_ids[i])
        #     acoustic_ids[i] = add_random_noise(acoustic_ids[i])
        #     acoustic[i] = add_random_noise(acoustic[i])
        #     visual[i] = add_random_noise(visual[i])
        outputs, kl_loss = model(
            input_ids,
            visual,
            acoustic,
            visual_ids,
            acoustic_ids,
            pos_ids, senti_ids, polarity_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
        )
        logits = outputs[0]
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), label_ids.view(-1)) + 1/(kl_loss+0.01)  # view张量维度变为1维
        lb = label_ids.reshape(global_configs.batch_size, 1)
        dif = lb - logits
        dif = torch.abs(dif)
        (input_ids_hard, visual_ids_hard, acoustic_ids_hard,
         pos_ids_hard, senti_ids_hard,
         polarity_ids_hard, visual_hard, acoustic_hard,
         input_mask_hard, segment_ids_hard, label_ids_hard) = hard_fea(dif, num_hard,
                                                                       input_ids,
                                                                       visual_ids,
                                                                       acoustic_ids,
                                                                       pos_ids,
                                                                       senti_ids,
                                                                       polarity_ids,
                                                                       visual,
                                                                       acoustic,
                                                                       input_mask,
                                                                       segment_ids,
                                                                       label_ids)
        selected_fea = [input_ids_hard, visual_ids_hard, acoustic_ids_hard,
         pos_ids_hard, senti_ids_hard,
         polarity_ids_hard, visual_hard, acoustic_hard,
         input_mask_hard, segment_ids_hard, label_ids_hard]
        # global hard_data_all
        # hard_data_all += hard_data
        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 梯度裁剪
        tr_loss += loss.item()
        nb_tr_steps += 1
        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.detach().cpu().numpy()
        logits = np.squeeze(logits).tolist()
        label_ids = np.squeeze(label_ids).tolist()  # squeeze 去掉维度为1的维度
        preds.extend(logits)  # extend 列表末尾追加另一个列表
        labels.extend(label_ids)

        # train hard

    preds = np.array(preds)
    labels = np.array(labels)

    return tr_loss / nb_tr_steps, preds, labels, selected_fea


def train_epoch_hard(args, model: nn.Module, selected_fea, optimizer, scheduler):
    model.train()
    preds = []
    labels = []
    tr_loss = 0
    optimizer.zero_grad()
    input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids, label_ids = selected_fea
    visual = torch.squeeze(visual, 1)
    outputs = model(
        input_ids,
        visual,
        acoustic,
        visual_ids,
        acoustic_ids,
        pos_ids, senti_ids, polarity_ids,
        attention_mask=input_mask,
        token_type_ids=segment_ids,
    )
    logits = outputs[0]
    loss_fct = MSELoss()
    loss = loss_fct(logits.view(-1), label_ids.view(-1))  # view张量维度变为1维
    if args.gradient_accumulation_step == 1:
        loss = loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 梯度裁剪
    tr_loss += loss.item()
    optimizer.step()
    scheduler.step()
    logits = logits.detach().cpu().numpy()
    label_ids = label_ids.detach().cpu().numpy()
    logits = np.squeeze(logits).tolist()
    label_ids = np.squeeze(label_ids).tolist()  # squeeze 去掉维度为1的维度
    preds.extend(logits)  # extend 列表末尾追加另一个列表
    labels.extend(label_ids)

        # train hard

    preds = np.array(preds)
    labels = np.array(labels)

    return tr_loss, preds, labels


def evaluate_epoch(args, model: nn.Module, dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []
    loss = 0
    nb_dev_examples, nb_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            outputs, kl_loss = model(
                input_ids,
                visual,
                acoustic,
                visual_ids,
                acoustic_ids,
                pos_ids, senti_ids, polarity_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
            logits = outputs[0]
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1)) + 1/(kl_loss+0.01)
            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step
            loss += loss.item()
            nb_steps += 1
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()
            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()
            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return loss / nb_steps, preds, labels


def train(
        args,
        model,
        train_dataloader,
        #validation_dataloader,
        test_data_loader,
        optimizer,
        scheduler, ):
    valid_losses = []
    test_accuracies = []
    test_accuracies2 = []
    for epoch_i in range(int(args.n_epochs)):
        train_loss, train_pre, train_label, selected_fea = train_epoch(args, model, train_dataloader, optimizer, scheduler)
        if hard_exam == True:
            _, _, _ = train_epoch_hard(args, model, selected_fea, optimizer, scheduler)
        #valid_loss, valid_pre, valid_label = evaluate_epoch(args, model, validation_dataloader)
        test_loss, test_pre, test_label = evaluate_epoch(args, model, test_data_loader)
        train_acc, train_mae, train_corr, train_f_score = score_model(train_pre, train_label)
        test_acc, test_mae, test_corr, test_f_score = score_model(test_pre, test_label)
        non0_test_acc, _, _, non0_test_f_score = score_model(test_pre, test_label, use_zero=True)
        #valid_acc, valid_mae, valid_corr, valid_f_score = score_model(valid_pre, valid_label)
        print(
            "epoch:{}, train_loss:{}, train_acc:{}, test_loss:{}, test_acc:{},test_mae:{}, test_corr:{}, test_f_score:{},non0_test_acc:{},non0_test_f_score:{}, learning_rate:{}".format(
                epoch_i, train_loss, train_acc, test_loss, test_acc, test_mae, test_corr, test_f_score, non0_test_acc,
                non0_test_f_score,
                optimizer.param_groups[0]['lr']
            )
        )
        #valid_losses.append(valid_loss)
        test_accuracies.append(test_acc)
        test_accuracies2.append(non0_test_acc)
        torch.save(model,
                       'D:/PyCharm Community Edition 2023.2.3/pythonProject/CENet/CENet-main/wandb/{}.pth'.format(test_acc))
        wandb.log(
            (
                {
                    "train_loss": train_loss,
          #          "valid_loss": valid_loss,
                    "train_acc": train_acc,
                    "train_corr": train_corr,
                  #  "valid_acc": valid_acc,
                   # "valid_corr": valid_corr,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_mae": test_mae,
                    "test_corr": test_corr,
                    "best_corr": np.max(test_corr),
                    "best_non0_test_acc": np.max(non0_test_f_score),
                    "best_test_f_score": np.max(test_f_score),
                    "best_mae": np.min(test_mae),
                    "best_test_acc": max(test_accuracies),
                    "best_test_acc2": max(test_accuracies2)
                }
            )
        )


def main():
    args = parser_args()
    # sweep_config = {
    #     'method': 'random'
    # }
    # metric = {
    #     'name': 'test_acc',
    #     'goal': 'maximize'
    # }
    # sweep_config['metric'] = metric
    # parameters_dict = {
    #     'dropout_prob': {
    #         'distribution': 'uniform',
    #         'min': 0.2,
    #         'max': 0.8
    #     },
    #     'learning_rate': {
    #         'distribution': 'log_uniform_values',
    #         'min': 6e-05,
    #         'max': 0.1
    #     },
    #     'weight_decay': {
    #         'distribution': 'uniform',
    #         'min': 0,
    #         'max': 1
    #     },
    #     'warmup_proportion': {
    #         'distribution': 'uniform',
    #         'min': 0,
    #         'max': 0.8
    #     },
    #     'adam_epsilon': {
    #         'distribution': 'log_uniform_values',
    #         'min': 1e-09,
    #         'max': 1e-06
    #     },
    # }
    # sweep_config['parameters'] = parameters_dict
    # sweep_id = wandb.sweep(sweep_config, project="CENet")
    wandb.init(project="CENet", reinit=True)

    set_random_seed(args.seed)
    wandb.config.update(args)

    (train_data_loader,
     #dev_data_loader,
     test_data_loader,
     num_train_optimization_steps,
     ) = set_up_data_loader(args)

    model, optimizer, scheduler = prep_for_training(args, num_train_optimization_steps)

    train(
        args,
        model,
        train_data_loader,
        #dev_data_loader,
        test_data_loader,
        optimizer,
        scheduler,
    )


if __name__ == "__main__":
    main()

# args = parser_args()
# config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=1, finetuning_task='sst')
# model = RobertaForSequenceClassification.from_pretrained(
#             args.model_name_or_path, config=config, pos_tag_embedding=True, senti_embedding=True, polarity_embedding=True)
# model.to(DEVICE)
# print(model)
