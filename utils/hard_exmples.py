import torch
import pickle
DEVICE = torch.device("cuda:0")
data_path = 'D:\PyCharm Community Edition 2023.2.3\pythonProject\CENet\CENet-jo\dataset\MOSI_16_sentilare_unaligned_data.pkl'

def top_k(loss, num):
    topk_values, topk_indices = torch.topk(loss, num, dim=0)
    return topk_indices

def hard_emp(loss, num, data):
    indices = top_k(loss, num)
    selected_data = [data[i] for i in indices]
    numpy_list = [tensor.cpu().detach().numpy() for tensor in selected_data]
    result = torch.tensor(numpy_list)
    result = torch.squeeze(result)
    return result.to(DEVICE)

def hard_fea(loss, num ,
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
             label_ids):
    input_ids_hard = hard_emp(loss, num, input_ids)
    visual_ids_hard = hard_emp(loss, num, visual_ids)
    acoustic_ids_hard = hard_emp(loss, num, acoustic_ids)
    pos_ids_hard = hard_emp(loss, num, pos_ids)
    senti_ids_hard = hard_emp(loss, num, senti_ids)
    polarity_hard = hard_emp(loss, num, polarity_ids)
    visual_hard = hard_emp(loss, num, visual)
    acoustic_hard = hard_emp(loss, num, acoustic)
    input_mask_hard = hard_emp(loss, num, input_mask)
    segment_ids_hard = hard_emp(loss, num, segment_ids)
    label_ids_hard = hard_emp(loss, num, label_ids)
    return input_ids_hard, visual_ids_hard, acoustic_ids_hard, pos_ids_hard, senti_ids_hard, polarity_hard, visual_hard, acoustic_hard, input_mask_hard, segment_ids_hard, label_ids_hard

