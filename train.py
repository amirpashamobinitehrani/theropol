import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import os
import argparse
import json
import random

from dataset import MouthData
from network import TheroNet
from utils import param_count


random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

def train(exp_path,
          log,
          optimization,
          loss):
    
    #tensorboard logger
    log_directory = os.path.join(log['directory'], exp_path)
    tb = SummaryWriter(os.path.join(log_directory, 'tensorboard'))

    #checkpoint directory
    ckpt_directory = os.path.join(log_directory, 'ckpt')
    if not os.path.isdir(ckpt_directory):
        os.makedirs(ckpt_directory)
        os.chmod(ckpt_directory, 0o775)
    print("ckpt_directory: ", ckpt_directory, flush=True)
    
    
    #load training data
    dataset = MouthData(**trainset_config)
    print("Data loaded.")
    
    #define model, print layers and number of parameters
    net = TheroNet(**network_config)
    
    #send the model to gpu
    if torch.cuda.is_available():
        net.cuda()
    print(net)
    print(f"TheröNet number of parameters: {param_count(net)}")

    #get loss and optimizer
    if loss["algorithm"] == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(net.parameters(), lr = optimization["learning_rate"] )
    loss = 0.0
    
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             shuffle = True, 
                                             batch_size = 8,
                                             num_workers = 4)
    
    
    if log["train_from_beginning"] == 0:
        ckpt_iter = -1
    
    n_iter = ckpt_iter + 1
    while n_iter < optimization["n_iters"] + 1:


        for images, labels in tqdm(dataloader, 0):
            
            optimizer.zero_grad()

            #forward and back-propagation
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 1e9)
            optimizer.step()

            #output to log
            loss += loss.item()
            if n_iter % log["iters_per_valid"] == 0:
                print("iteration: {} \ loss: {:.7f}".format(
                    n_iter, loss.item()), flush=True)

                tb.add_scalar("Train/Train-Loss", loss.item(), n_iter)
                tb.add_scalar("Train/Gradient-Norm", grad_norm, n_iter)
            
            #save checkpoints
            if n_iter > 0 and n_iter % log["iters_per_ckpt"] == 0:
                checkpoint_name = "{}_{}.pkl".format("TheroNet", n_iter)
                torch.save({'iter': n_iter,
                           'model_state_dict': net.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(ckpt_directory, checkpoint_name))
                print('model at saved at iteration %s' % n_iter)

            n_iter += 1
    
    return 0


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/Users/amirpashamobinitehrani/Desktop/TheroPol/MouthNet/script/config/theronet.json')
    args = parser.parse_args()
    
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    
    train_config = config["train"]          # training parameters
    global network_config
    network_config = config["network"]      # to define network
    global trainset_config
    trainset_config = config["trainset"]    # to load trainset

    train(**train_config)
