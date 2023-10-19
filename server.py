import pickle
import numpy
from collections import OrderedDict
import zmq
import sys
import threading
import time
from random import randint, random
import time
import zmq

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import warnings

warnings.filterwarnings('ignore')
numpy.set_printoptions(suppress=False)
torch.set_printoptions(sci_mode=False)
import tenseal as ts
import pickle

################################################################
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torchsummary import summary
import numpy as np
import torchvision.utils as vutils
from torch import nn, optim
from torch.nn import functional as F

ngpu = 1


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.GroupNorm(64, 64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.GroupNorm(64, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 1, kernel_size=(4, 4), stride=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            nn.ConvTranspose2d(100, 128, 4, 1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 3, 2, 1, bias=False),
            nn.GroupNorm(32, 64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.GroupNorm(32, 32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


G = Generator(ngpu)
D = Discriminator(ngpu)

# print(G)

################################################################

# D.load_state_dict(torch.load('Dis.ckpt'))
discrimiator = D.state_dict()

# for k, v in discrimiator.items():
#     print(k)


# G.load_state_dict(torch.load('Gen.ckpt'))
generator = G.state_dict()

# for k, v in generator.items():
#     print(k)

vals111 = []
keys = []
vals = []
for k, v in discrimiator.items():
    keys.append(k)
    vals.append(v)
    a = v.numpy()
    vals111.append(a.shape)

# print(vals111)
# vals111 = numpy.array(vals)
# keys = numpy.array(keys)
# vals = numpy.array(vals)


keys1 = []
vals1 = []
for k1, v1 in generator.items():
    keys1.append(k1)
    vals1.append(v1)


# keys1 = numpy.array(keys1)
# vals1 = numpy.array(vals1)


# print("Model's state_dict:")
# for param_tensor in model2.state_dict():
#     print(torch.numel(model2.state_dict()[param_tensor]))


###################################################################
def elapsed_time_total(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        "Total Traning Time: {:0>2}:{:0>2}:{:05.2f}"
        .format(int(hours), int(minutes), seconds)
        )


###################################################################


import tenseal as ts

import base64

# Assuming UTF-8 encoding, change to something else if you need to
base64.b64encode("password".encode("utf-8"))


def write_data(file_name, data):
    if type(data) == bytes:
        # bytes to base64
        data = base64.b64encode(data)

    with open(file_name, 'wb') as f:
        f.write(data)


def read_data(file_name):
    with open(file_name, "rb") as f:
        data = f.read()
    # base64 to bytes
    return base64.b64decode(data)


###################################################################


global data_list
global client_num

client_num = 0

context = zmq.Context()
socket = context.socket(zmq.ROUTER)
socket.bind("tcp://*:5555")

pub_socket = context.socket(zmq.PUB)
pub_socket.bind("tcp://*:5557")

start_total = time.time()

print("The server is running now!")

c = 0
data_list = []
loaded_enc = []
loaded_enc_tmp = []
cipher1 = []
cipher2 = []
sum_ = 0

data_list_dicriminator = []
data_list_generator = []
sum_1 = 0
sum_2 = 0
count = 1
while c < 39:

    #     print(G)
    ident1, msg1 = socket.recv_multipart()
    ident2, msg2 = socket.recv_multipart()

    string = b"New"

    if string == msg1 and string == msg2:

        client_num = client_num + 1

        message1 = pickle.dumps(vals)
        socket.send_multipart([ident1, message1])

        message2 = pickle.dumps(vals1)
        socket.send_multipart([ident2, message2])

        print("Base model sent to the new client!")

    else:

        print("Training round started")

        message1 = pickle.loads(msg1)
        message2 = pickle.loads(msg2)

        if len(message1) == 8:

            data_list_dicriminator.append(message1)
        else:
            data_list_generator.append(message1)

        if len(message2) == 8:
            data_list_dicriminator.append(message2)
        else:
            data_list_generator.append(message2)

        if len(data_list_dicriminator) == 2 and len(data_list_generator) == 2:

            print("Enough data recevied")

            print("len(data_list_dicriminator): ", len(data_list_dicriminator))
            print("len(data_list_generator): ", len(data_list_generator))

            loaded_context = ts.context_from(read_data("public.txt"))

            for i in range(len(data_list_dicriminator)):
                for j in range(len(data_list_dicriminator[0])):
                    loaded_enc_dicriminator = ts.ckks_tensor_from(loaded_context, data_list_dicriminator[i][j])
                    seri_loaded_dis = loaded_enc_dicriminator.serialize()
                    write_data(f"loaded_enc_dicriminator{i}{j}.txt", seri_loaded_dis)

            print("dicriminator encrypted loaded")

            for i in range(len(data_list_generator)):
                for j in range(len(data_list_generator[0])):
                    loaded_enc_generator = ts.ckks_tensor_from(loaded_context, data_list_generator[i][j])
                    seri_loaded_gen = loaded_enc_generator.serialize()
                    write_data(f"loaded_enc_generator{i}{j}.txt", seri_loaded_gen)

            print("generator encrypted loaded")

            for i in range(len(data_list_dicriminator[0])):
                for j in range(len(data_list_dicriminator)):
                    enc_read_dis = read_data(f"loaded_enc_dicriminator{j}{i}.txt")
                    loaded_enc_dis = ts.ckks_tensor_from(loaded_context, enc_read_dis)
                    sum_1 += loaded_enc_dis

                sum_1 = sum_1 * 0.25
                sum_final1 = sum_1.serialize()
                cipher1.append(sum_final1)
                sum_1 = 0
                sum_final1 = 0

            print("Avgg dicriminator encrypted computed")

            for i in range(len(data_list_generator[0])):
                for j in range(len(data_list_generator)):
                    enc_read_gen = read_data(f"loaded_enc_generator{j}{i}.txt")
                    loaded_enc_gen = ts.ckks_tensor_from(loaded_context, enc_read_gen)
                    sum_2 += loaded_enc_gen
                sum_2 = sum_2 * 0.25
                sum_final2 = sum_2.serialize()
                cipher2.append(sum_final2)
                sum_2 = 0
                sum_final2 = 0

            print("Avgg generator encrypted computed")

            message1 = pickle.dumps(cipher1)
            message2 = pickle.dumps(cipher2)

            pub_socket.send(message1)
            pub_socket.send(message2)

            print("Sent!")
            print("Round: ", count)
            count = count + 1

            cipher1 = []
            cipher2 = []
            sum_1 = 0
            sum_2 = 0
            sum_final1 = 0
            sum_final2 = 0
            data_list_dicriminator = []
            data_list_generator = []

        c = c + 1

end_total = time.time()
elapsed_time_total(start_total, end_total)