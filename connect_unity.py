
import numpy as np
from rmi.model.network import Decoder, Discriminator, InputEncoder, LSTMNetwork

import wandb
from tqdm import tqdm
import datetime
import random
import torch
import socket





def connectUnity3d(state_encoder, lstm, decoder):
    count = 0
    
    # connect to Unity
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 5000)
    server_socket.bind(server_address)
    server_socket.listen(1)
    print('Waiting for connection ...')
    client_socket, client_address = server_socket.accept()
    print('Successful connection：', client_address)
    
    input_poses = None
    toUnity_poses = None
    while True:
        # receive from Unity
        receive_bytes = client_socket.recv(50000)
        fromUnity = unserialized_to_tensor(receive_bytes)
        input_poses = fromUnity.unsqueeze(0).unsqueeze(0)
        
        target_space, world_space, cross_target_space, target = split(input_poses)
        
        state_input = target_space[:,-1].clone().detach()
        
        # encoder
        h_state = state_encoder(state_input.clone().detach())
        
        # lstm
        h_in = h_state.unsqueeze(0)
        h_out = lstm(h_in)

        # decoder
        h_pred = decoder(h_out.squeeze(0))
        toUnity_poses = h_pred[0]
        
        send_bytes = serialize_from_tensor(toUnity_poses)
        # print(f"toUnity: {toUnity.shape}")
        client_socket.sendall(send_bytes)
        count += 1
        
        
        
def split(pose):
    char1, char2 = pose[:,:,:234*3+240], pose[:,:,234*3+240:]
    # target_space = torch.cat([char1[:,:,:234], char2[:,:,:234]], dim=2)
    target_space = torch.cat([torch.zeros(1,1,6).cuda(), char1[:,:,:234], torch.zeros(1,1,6).cuda(), char2[:,:,:234]], dim=2)
    world_space = torch.cat([char1[:,:,234:234*2], char2[:,:,234:234*2]], dim=2)
    cross_target_space = torch.cat([char1[:,:,234*2:234*3], char2[:,:,234*2:234*3]], dim=2)
    target = torch.cat([char1[:,-1:,234*3:234*3+240], char2[:,-1:,234*3:234*3+240]], dim=2)
    return target_space, world_space, cross_target_space, target

def unserialized_to_tensor(data):
    data = str(data).replace('b','').replace("'","").split(sep=' ')
    while '' in data:
        data.remove('')
    data = np.array(data).astype(float)
    data = torch.from_numpy(data).cuda().float()
    return data

def serialize_from_tensor(data):
    features = data.shape[0]
    data = data.tolist()
    toUnity = ''
    for i in range(features):
        toUnity += str(round(data[i], 5)) + ","
    
    return toUnity.encode()


if __name__ == '__main__':
    # Initializing networks
    state_in = 240*2 # root_v_dim + local_q_dim + contact_dim
    state_encoder = InputEncoder(input_dim=state_in)
    state_encoder.cuda()

    # LSTM
    lstm_in = state_encoder.out_dim #* 3
    lstm = LSTMNetwork(input_dim=lstm_in, hidden_dim=lstm_in).cuda()
    lstm.cuda()

    # Decoder
    decoder = Decoder(input_dim=lstm_in, out_dim=state_in)
    decoder.cuda()
    
    ckpt_path = "D:/Robust-Motion-In-betweening/checkpoint/rmi/ckpt_last.pth.tar"
    state_encoder.load_state_dict(torch.load(ckpt_path)['state_encoder'])
    lstm.load_state_dict(torch.load(ckpt_path)['lstm'])
    decoder.load_state_dict(torch.load(ckpt_path)['decoder'])
    
    lstm.init_hidden(1)
    
    connectUnity3d(state_encoder, lstm, decoder)