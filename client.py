from __future__ import print_function

import random

import numpy
import pickle
from collections import OrderedDict
import json
import zmq
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy
import sys
from uuid import uuid4
import uuid
import warnings

warnings.filterwarnings("ignore")
numpy.set_printoptions(suppress=False)
torch.set_printoptions(sci_mode=False)
torch.set_num_threads(5)
# import tenseal as ts
import tenseal.sealapi as sealapi
from damgard_jurik import PrivateKeyRing
import pickle
import time
from tensor import EncryptedTensor

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


def is_pickle_stream(stream):
    try:
        pickle.loads(stream)
        return True
    except:
        return False


context2 = zmq.Context()
print("Connecting to key server ...")
socket = context2.socket(zmq.DEALER)
socket.connect("tcp://localhost:5555")
identity = str(id)
socket.identity = identity.encode("ascii")

context1 = zmq.Context()
print("Connecting to hello world server…")
socket1 = context1.socket(zmq.DEALER)
socket1.connect("tcp://localhost:5556")
identity = str(id)
socket1.identity = identity.encode("ascii")

sub_socket = context2.socket(zmq.SUB)
sub_socket.connect("tcp://localhost:5557")
sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")


def train(discriminator_arrived, generator_arrived):
    import random
    import numpy as np
    import torch.backends.cudnn as cudnn
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils.data
    from torchvision import datasets, transforms
    import torchvision.transforms as transforms
    import torchvision.utils as vutils
    from opacus import PrivacyEngine
    from tqdm import tqdm
    import matplotlib

    matplotlib.use("module://matplotlib-backend-kitty")
    import matplotlib.pyplot as plt
    import time

    nc = 1

    workers = 2
    batch_size = 128
    imageSize = 28

    epochs = 1
    lr = 0.0002
    beta1 = 0.5
    ngpu = 1

    target_digit = 9

    nc = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    disable_dp = True

    secure_rng = False

    r = 1

    n_runs = 1

    sigma = 0.5

    max_per_sample_grad_norm = 1.0

    delta = 1e-4

    nz = 100

    def elapsed_time(start, end):
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            "Elapsed Time: {:0>2}:{:0>2}:{:05.2f}".format(
                int(hours), int(minutes), seconds
            )
        )

    def elapsed_time_total(start, end):
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            "Total Traning Time: {:0>2}:{:0>2}:{:05.2f}".format(
                int(hours), int(minutes), seconds
            )
        )

    manualSeed = random.randint(1, 50000)

    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    try:
        dataset = datasets.MNIST(
            root="PP-FEDGAN/Data",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )

    except ValueError:
        print("Cannot load dataset")

    dataset_range = list(range(0, 5000))

    trainset_range = torch.utils.data.Subset(dataset, dataset_range)

    dataloader = torch.utils.data.DataLoader(
        trainset_range,
        num_workers=int(workers),
        batch_size=batch_size,
    )

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def elapsed_time(start, end):
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            "Elapsed Time: {:0>2}:{:0>2}:{:05.2f}".format(
                int(hours), int(minutes), seconds
            )
        )

    def elapsed_time_total(start, end):
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            "Total Traning Time: {:0>2}:{:0>2}:{:05.2f}".format(
                int(hours), int(minutes), seconds
            )
        )

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
    G = G.to(device)

    D = Discriminator(ngpu)
    D = D.to(device)

    fixed_noise = torch.randn(batch_size, 100, 1, 1, device=device)

    G.apply(weights_init)
    D.apply(weights_init)

    REAL_LABEL = 1.0
    FAKE_LABEL = 0.0
    #######################################################

    discriminator_old = D.state_dict()
    shape_d = []
    for name, param in discriminator_old.items():
        shape_d.append(param.cpu().numpy())

    #     discriminator_arrived, generator_arrived

    valss_d = []
    de_ser_d = []

    #     if isinstance(discriminator_arrived[0], (bytes, bytearray)):
    #         print("byte")
    #         for i in range(len(discriminator_arrived)):
    #             loaded_enc_dis_1 = ts.ckks_tensor_from(context, discriminator_arrived[i])
    #             final_dd = (loaded_enc_dis_1.decrypt().tolist())
    #             print(len(final_dd))

    print("Starting decryption of received model")
    decrypt_time = time.time()

    # Decode EncryptedTensors to Tensors given the private keys for both models
    if isinstance(discriminator_arrived[0], EncryptedTensor):
        valss_d = [
            d.to_tensor(private_key_ring=private_ring) for d in discriminator_arrived
        ]
    elif isinstance(discriminator_arrived[0], torch.Tensor):
        valss_d = discriminator_arrived
    else:
        raise ValueError(f"Unexpected type {type(discriminator_arrived[0])}")

    generator_old = G.state_dict()
    shape_g = []

    for name, param in generator_old.items():
        shape_g.append(param.cpu().numpy())

    valss_g = []
    de_ser_g = []

    if isinstance(generator_arrived[0], (EncryptedTensor)):
        valss_g = [
            g.to_tensor(private_key_ring=private_ring) for g in generator_arrived
        ]
    elif isinstance(generator_arrived[0], torch.Tensor):
        valss_g = generator_arrived
    else:
        raise ValueError(f"Unexpected type {type(generator_arrived[0])}")

    print("--- Decryption finished in %s seconds ---" % (time.time() - decrypt_time))

    #######################################################
    i = 0

    for name, param in discriminator_old.items():
        # Don't update if this is not a weight.
        #     if not "weight" in name:
        #         continue

        # Transform the parameter as required.
        transformed_param_d = valss_d[i]

        # Update the parameter.
        param.copy_(transformed_param_d)
        i = i + 1

    j = 0

    for name, param in generator_old.items():
        # Don't update if this is not a weight.
        #     if not "weight" in name:
        #         continue

        # Transform the parameter as required.
        transformed_param_g = valss_g[j]

        # Update the parameter.
        param.copy_(transformed_param_g)
        j = j + 1

        #######################################################

    criterion = nn.BCELoss()

    optim_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

    if not disable_dp:
        privacy_engine = PrivacyEngine()

        D, optim_D, dataloader = privacy_engine.make_private(
            module=D,
            optimizer=optim_D,
            data_loader=dataloader,
            noise_multiplier=sigma,
            max_grad_norm=max_per_sample_grad_norm,
        )

    optim_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

    start_total = time.time()
    iters = 0
    img_list = []
    fixed_noise1 = torch.randn(16, 100, 1, 1)

    for epoch in range(epochs):
        start = time.time()

        data_bar = tqdm(dataloader)

        for i, data in enumerate(data_bar, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            optim_D.zero_grad(set_to_none=True)

            real_data = data[0].to(device)
            batch_size = real_data.size(0)

            # train with real
            label_true = torch.full((batch_size,), REAL_LABEL, device=device)
            output = D(real_data)
            errD_real = criterion(output, label_true)
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = G(noise)
            label_fake = torch.full((batch_size,), FAKE_LABEL, device=device)
            output = D(fake.detach())
            errD_fake = criterion(output, label_fake)

            # below, you actually have two backward passes happening under the hood
            # which opacus happens to treat as a recursive network
            # and therefore doesn't add extra noise for the fake samples
            # noise for fake samples would be unnecesary to preserve privacy

            errD = errD_real + errD_fake
            errD.backward()
            optim_D.step()
            optim_D.zero_grad(set_to_none=True)

            D_G_z1 = output.mean().item()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            optim_G.zero_grad()

            label_g = torch.full((batch_size,), REAL_LABEL, device=device)
            output_g = D(fake)
            errG = criterion(output_g, label_g)
            errG.backward()
            D_G_z2 = output_g.mean().item()
            optim_G.step()

            if not disable_dp:
                epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(
                    delta=delta
                )
                data_bar.set_description(
                    f"epoch: {epoch}, Loss_D: {errD.item()} "
                    f"Loss_G: {errG.item()} D(x): {D_x} "
                    f"D(G(z)): {D_G_z1}/{D_G_z2}"
                    "(ε = %.2f, δ = %.2f) for α = %.2f" % (epsilon, delta, best_alpha)
                )
            else:
                data_bar.set_description(
                    f"epoch: {epoch}, Loss_D: {errD.item()} "
                    f"Loss_G: {errG.item()} D(x): {D_x} "
                    f"D(G(z)): {D_G_z1}/{D_G_z2}"
                )

        from imutils import build_montages
        import os
        import cv2

        #         benchmarkNoise = torch.randn(256, 100, 1, 1, device=device)

        # set the generator in evaluation phase, make predictions on
        # the benchmark noise, scale it back to the range [0, 255],
        # and generate the montage

        from random import randrange

        c = randrange(1000)

        #         if (iters % 500 == 0):

        #             G.eval()
        #             images = G(benchmarkNoise)
        #             images = images.detach().cpu().numpy().transpose((0, 2, 3, 1))
        #             images = ((images * 127.5) + 127.5).astype("uint8")
        #             images = np.repeat(images, 3, axis=-1)
        #             vis = build_montages(images, (64, 64), (8, 4))[0]
        #             p = os.path.join('/content', "epoch_{}_{}.png".format(str(epoch + 1).zfill(4),c))
        #             cv2.imwrite(p, vis)

        with torch.no_grad():
            fake = G(fixed_noise1).to(device).cpu().detach()
        img_list.append(vutils.make_grid(fake, nrow=8, normalize=True))

        iters += 1

        end = time.time()
        elapsed_time(start, end)

        fig = plt.figure(figsize=(5, 5))
        rand_noise = torch.rand((64, 100, 1, 1))
        out = vutils.make_grid(
            G(rand_noise.to(device)).cpu().detach(), padding=5, normalize=True
        )
        plt.imshow(np.transpose(out.numpy(), (1, 2, 0)), cmap="gray")
        plt.show()

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 2)
    plt.axis("off")
    #     plt.title(sigma)
    #     plt.title('Scatter plot pythonspot.com', y=-0.01)
    plt.imshow(np.transpose(img_list[-1].numpy(), (1, 2, 0)))
    plt.show()

    print(iters)

    #     from IPython.display import Image
    #     from torchvision.utils import save_image
    #     import os

    #     sample_vectors = torch.rand(100, 100, 1, 1).to(device)
    #     sample_dir = "/MNIST_DP_TEST/Data/1.4/"

    #     def denorm(x):
    #         out = (x + 1) / 2
    #         return out.clamp(0, 1)

    #     def save_fake_images(index, fixed_noise11):
    #         fake_images = G(fixed_noise11)
    #         fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    #         fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    #         print('Saving', fake_fname)
    #         save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=10)

    #     for i in range (600):
    #         torch.manual_seed(i)
    #         fixed_noise11 = torch.randn(100, 100, 1, 1).cuda()
    #     # Before training
    #         save_fake_images(i,fixed_noise11)
    #     # Image(os.path.join(sample_dir, 'fake_images-0000.png'))

    #######################################################

    model_last_d = D.state_dict()

    keys_d = []
    vals_d = []

    for k, v in model_last_d.items():
        keys_d.append(k)
        vals_d.append(v.cpu())

    #     keys_d = numpy.array(keys_d)
    #     vals_d = numpy.array(vals_d)

    start_time = time.time()
    cipher_d = [EncryptedTensor.Create(tensor=val, public_key=public) for val in vals_d]
    # for i in range(len(vals_d)):
    #     cipher_d.append(EncryptedTensor(tensor=vals_d[i], public_key=public))

    # print(vals_d[i].shape)
    # encrypted_tensor_d = ts.ckks_tensor(context, vals_d[i], batch=True)
    # print(encrypted_tensor_d)
    # ser_tensor_d = encrypted_tensor_d.serialize()
    # print(len(ser_tensor_d))
    # cipher_d.append(ser_tensor_d)
    # print(len(cipher_d))

    model_last_g = G.state_dict()

    keys_g = []
    vals_g = []

    for h, z in model_last_g.items():
        keys_g.append(h)
        vals_g.append(z.cpu())

    #     keys_g = numpy.array(keys_g)
    #     vals_g = numpy.array(vals_g)

    cipher_g = [EncryptedTensor.Create(tensor=val, public_key=public) for val in vals_g]
    # for i in range(len(vals_g)):
    #     cipher_g.append(EncryptedTensor(tensor=vals_g[i], public_key=public))

    # encrypted_tensor_g = ts.ckks_tensor(context, vals_g[i], batch=True)
    # ser_tensor_g = encrypted_tensor_g.serialize()
    # cipher_g.append(ser_tensor_g)

    print("--- Encryption finished in %s seconds ---" % (time.time() - start_time))

    #######################################################

    end_total = time.time()
    elapsed_time_total(start_total, end_total)

    return cipher_d, cipher_g


# context = ts.context_from(read_data("secret.txt"))
# load keys
public = pickle.loads(read_data("key.pub"))
private_ring: PrivateKeyRing = pickle.loads(read_data("key"))

# remove random share to keep 2/3, manually re-init PrivateKeyRing
private_ring.private_key_shares.pop(
    random.randrange(len(private_ring.private_key_shares))
)
private_ring.i_list = [pks.i for pks in private_ring.private_key_shares]
private_ring.S = set(private_ring.i_list)

print(f"Received keys: {len(private_ring.private_key_shares)} private key shares")

for request in range(10):
    print("request in loop: ", request)

    if request == 0:
        print("request in if", request)
        message = b"New"
        socket.send(message)
        socket.send(message)

        discriminator = socket.recv()
        generator = socket.recv()
        discriminator = pickle.loads(discriminator)
        generator = pickle.loads(generator)

    else:
        print("request in else", request)
        discriminator = sub_socket.recv()
        generator = sub_socket.recv()
        print(f"New update number {request} received from the server ")

    #     ident,  message = socket.recv_multipart()
    if is_pickle_stream(discriminator) and is_pickle_stream(generator):
        discriminator = pickle.loads(discriminator)
        generator = pickle.loads(generator)
        print(f"if train started")
        print(type(discriminator[0]))
        print(len(discriminator))
        cipher_d, cipher_g = train(discriminator, generator)
        print("len(cipher_d):   ", len(cipher_d))
        print("len(cipher_g):   ", len(cipher_g))
    else:
        print(f"else train started: {type(discriminator)} {type(generator)}")
        cipher_d, cipher_g = train(discriminator, generator)
        print("len(cipher_d):   ", len(cipher_d))
        print("len(cipher_g):   ", len(cipher_g))
    print("train finished")
    print(f"Model number {request} locally trained")

    cipher_d_ = pickle.dumps(cipher_d)
    cipher_g_ = pickle.dumps(cipher_g)
    socket.send(cipher_d_)
    socket.send(cipher_g_)
