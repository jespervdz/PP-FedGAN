import os
import time
import warnings

import numpy
import torch
import zmq

warnings.filterwarnings("ignore")
numpy.set_printoptions(suppress=False)
torch.set_printoptions(sci_mode=False)

from damgard_jurik import keygen

import pickle

################################################################
from torch import nn

ngpu = 1
n_clients = 3


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
            nn.Tanh(),
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
        "Total Traning Time: {:0>2}:{:0>2}:{:05.2f}".format(
            int(hours), int(minutes), seconds
        )
    )


###################################################################


import base64

# Assuming UTF-8 encoding, change to something else if you need to
base64.b64encode("password".encode("utf-8"))


def write_data(file_name, data):
    if type(data) == bytes:
        # bytes to base64
        data = base64.b64encode(data)

    with open(file_name, "wb") as f:
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

# generate and save keys
public, private_ring = keygen(n_bits=64, s=1, threshold=2, n_shares=3)
write_data("key.pub", pickle.dumps(public))
write_data("key", pickle.dumps(private_ring))

print("Generated and saved keys")

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

        socket.send_string("test")

        print("Base model sent to the new client!")

    else:
        print("Training round started")

        message1 = pickle.loads(msg1)
        message2 = pickle.loads(msg2)

        print(type(message1))

        if len(message1) == 8:
            data_list_dicriminator.append(message1)
        else:
            data_list_generator.append(message1)

        if len(message2) == 8:
            data_list_dicriminator.append(message2)
        else:
            data_list_generator.append(message2)

        if (
            len(data_list_dicriminator) == n_clients
            and len(data_list_generator) == n_clients
        ):
            print("Enough data recevied")

            print("len(data_list_dicriminator): ", len(data_list_dicriminator))
            print("len(data_list_generator): ", len(data_list_generator))

            for tensor in range(len(data_list_dicriminator[0])):
                res = data_list_dicriminator[0][tensor]
                for client in range(1, len(data_list_dicriminator)):
                    loaded = data_list_dicriminator[client][tensor]
                    res = res + loaded
                    # debug
                    # loaded.to_tensor(private_key_ring=private_ring)
                cipher1.append(res)

            print("dicriminator encrypted loaded")

            for tensor in range(len(data_list_generator[0])):
                res = data_list_generator[0][tensor]
                for client in range(1, len(data_list_generator)):
                    loaded = data_list_generator[client][tensor]
                    res += loaded
                cipher2.append(res)

            print("generator encrypted loaded")

            print("Avgg generator encrypted computed")

            for item in cipher1:
                print(type(item))
            message1 = pickle.dumps(cipher1)  # list of encrypted tensors
            message2 = pickle.dumps(cipher2)  # list of encrypted tensors

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

os.remove("key")
os.remove("key.pub")
