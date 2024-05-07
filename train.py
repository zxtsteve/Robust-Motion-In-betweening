import os
import pathlib

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.data.utils import flip_bvh
from rmi.model.network import Decoder, Discriminator, InputEncoder, LSTMNetwork
from rmi.model.noise_injector import noise_injector
from rmi.model.positional_encoding import PositionalEncoding
from rmi.model.skeleton import (Skeleton, amass_offsets, sk_joints_to_remove,
                                sk_offsets, sk_parents)
import shutil

def train():
    # Data
    # data = read_data("D:/MyData/Boxing/1target+poses/len70_random-40+0_pose_in_target_joint", num_seq=100)
    data = read_data("/home2/sgtn88/datasets/Boxing/len70_random-40+0_target_joint_space", num_seq=30000)

    # Initializing networks
    state_in = 234*2 # root_v_dim + local_q_dim + contact_dim
    state_encoder = InputEncoder(input_dim=state_in)
    state_encoder.cuda()

    # offset_in = 78*2 # root_v_dim + local_q_dim
    # offset_encoder = InputEncoder(input_dim=offset_in)
    # offset_encoder.cuda()

    # target_in = 234*2 # local_q_dim
    # target_encoder = InputEncoder(input_dim=target_in)
    # target_encoder.cuda()

    # LSTM
    lstm_in = state_encoder.out_dim #* 3
    lstm = LSTMNetwork(input_dim=lstm_in, hidden_dim=lstm_in).cuda()
    lstm.cuda()

    # Decoder
    decoder = Decoder(input_dim=lstm_in, out_dim=state_in)
    decoder.cuda()


    generator_optimizer = Adam(params=list(state_encoder.parameters()) + 
                                    #   list(offset_encoder.parameters()) + 
                                    #   list(target_encoder.parameters()) +
                                      list(lstm.parameters()) +
                                      list(decoder.parameters()))

    mse = torch.nn.MSELoss()
    
    state_encoder.train()
    # offset_encoder.train()
    # target_encoder.train()
    lstm.train()
    decoder.train()

    batch_size = 32
    for epoch in tqdm(range(100)):
        loss_epoch = 0
        for b in range(0, len(data), batch_size):
            _data = torch.cat([torch.tensor(s).unsqueeze(0) for s in data[b:b+batch_size]], dim=0).clone().detach()
            _data = torch.cat([_data[:,:,6:240], _data[:,:,246:480]], dim=2)
            current_batch_size = _data.shape[0]
            lstm.init_hidden(current_batch_size)
            loss_total = 0
            for f in range(1, 71-1):
                # state_input = data[b:b+batch_size, f, :state_in].clone().detach()
                # offset_input = data[b:b+batch_size, f, state_in:state_in+offset_in].clone().detach()
                # target_input = data[b:b+batch_size, f, -target_in:].clone().detach()
                
                state_input = _data[:, f].clone().detach()
                
                h_state = state_encoder(state_input.clone().detach())
                # h_offset = offset_encoder(offset_input.clone().detach())
                # h_target = target_encoder(target_input.clone().detach())
                

                # lstm
                # h_in = torch.cat([h_state, h_offset, h_target], dim=1).unsqueeze(0)
                h_in = h_state.unsqueeze(0)
                h_out = lstm(h_in)

                # decoder
                h_pred = decoder(h_out.squeeze(0))

                # loss
                # gt = data[b:b+batch_size, f+1].clone().detach()
                gt = _data[:, f+1].clone().detach()
                # loss_total = mse(h_pred, gt)
                loss_total += torch.mean(torch.norm(h_pred - gt, dim=1))
                
            loss_epoch += loss_total.item() * current_batch_size

            generator_optimizer.zero_grad()
            loss_total.backward()
            generator_optimizer.step()

        print("loss: ", loss_epoch / len(data) / 70)
        save_ckpt("rmi", state_encoder, lstm, decoder)


def read_data(path, num_seq=0):
    print(f"Reading data from {path}")
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files = [f for f in files if "txt" in f]
    if num_seq!=0: files = files[:num_seq]
    sequences = []
    idx = 0
    for file in files:
        filename = path + '/' + file
        seq = read_text(filename)
        seq = torch.from_numpy(seq).float().cuda().requires_grad_(False) # (t,c)
        sequences.append(seq) # [(t,c)]
        idx += 1
    assert all(seq.shape[1]==sequences[0].shape[1] for seq in sequences), "Not all seq has same channels"
    return sequences

def read_text(filename):
    returnArray = []
    with open(filename) as file:
        try:
            for line in file.readlines():
                data = line.strip('/n').split()
                for i in range(len(data)):
                    data[i] = float(data[i])
                returnArray.append(data)
        except:
            print(filename+" 文件出错")

    returnArray = np.array(returnArray)
    return returnArray

def save_ckpt(name, state_encoder, lstm, decoder):
    path = f"./checkpoint/{name}"
    if not os.path.exists(path): os.makedirs(path)
    
    ckpt_dict = {'state_encoder': state_encoder.state_dict(),
                #  'offset_encoder': offset_encoder.state_dict(),
                #  'target_encoder': target_encoder.state_dict(),
                 'lstm': lstm.state_dict(),
                 'decoder': decoder.state_dict()}
    
    ckpt_path = path + "/ckpt_last.pth.tar"
    torch.save(ckpt_dict, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    
    return 

if __name__ == '__main__':
    train()