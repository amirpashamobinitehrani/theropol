import torch
import json
import os
import argparse
from network import TheroNet


#load model
def load_model(path):
    net = TheroNet(**network_config)
    ckpt = torch.load(path)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    return net


def create_tesor(image_size, grayscale=True):
    if grayscale:
        x = torch.randn((1, 1, image_size, image_size))
    else:
        x = torch.randn((1, 3, image_size, image_size))
    return x


def to_script(model, input, path):
    scripted = torch.jit.trace(model, input)
    save_path = os.path.join(path, "TheroNet.ts")
    scripted.save(save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-p', '--model_path', type=str)
    parser.add_argument('-t', '--script_path', type=str)
    args = parser.parse_args()
    
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)

    global network_config
    network_config = config["network"]   # to define network

    model = load_model(args.model_path)
    x = create_tesor(image_size=128)
    
    #convert to torch script
    to_script(model, x, args.script_path)

