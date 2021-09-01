import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader

import numpy as np
import scipy.io as sio
import os
from dataset import dataset, dataset_full

def main():
    epoch = 200
    learning_rate = 1e-4
    cs_ratio = 1
    total_layer = 9
    group_num = 1
    gpu_list = '0'

    data_dir = 'data'
    matrix_dir = 'sampling_matrix'
    model_dir = 'model'
    log_dir = 'log'

    n_input = round(1089 * cs_ratio / 100)
    n_output = 1089
    n_train = 88912
    batch_size = 64
    
    train_name = 'Training_Data.mat'
    train_data_name = './{folder_name}/{name}'.format(folder_name = data_dir, name = train_name)
    sampling_folder_name = './{folder_name}'.format(folder_name = matrix_dir)
    sampling_matrix_name = './{folder_name}/phi_0_{ratio}_1089_random.mat'.format(folder_name = matrix_dir, ratio = cs_ratio)
    log_folder_name = './{folder_name}'.format(folder_name = log_dir)
    log_file_name = './{folder_name}/Log_AMP_Net_layer_{layer}_group_{group}_ratio_{ratio}_lr_{lr:.4f}.txt'.format(
        folder_name = log_dir, layer = total_layer, group = group_num, ratio = cs_ratio, lr = learning_rate)
    model_folder_name = './{folder_name}/AMP_Net_layer_{layer}_group_{group}_ratio_{ratio}_lr_{lr:.4f}'.format(
        folder_name = model_dir, layer = total_layer, group = group_num, ratio = cs_ratio, lr = learning_rate)

    if not os.path.exists(sampling_folder_name):
        os.mkdir(sampling_folder_name)
    if not os.path.exists(log_folder_name):
        os.mkdir(log_folder_name)
    if not os.path.exists(model_folder_name):
        os.mkdir(model_folder_name)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sampling_matrix = torch.randn(n_input, n_output) * 0.01
    sampling_matrix_np = sampling_matrix.numpy()
    sampling_matrix = sampling_matrix.to(device)
    sio.savemat(sampling_matrix_name, {'sampling_matrix': sampling_matrix_np})

    net = AMPNet(total_layer, sampling_matrix)
    net = nn.DataParallel(net)
    net = net.to(device)

    train_dataset = dataset(train = True, transform = None, target_transform = None)
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    optimizer = optim.Adam(net.parameters(), lr = learning_rate)

    net.train()
    for num in range(epoch):
        for data, _ in train_loader:
            data = data.float()
            data = data.to(device)
            data = torch.unsqueeze(data, dim = 1)

            output = net(data)
            data = torch.squeeze(data, dim = 1)
            loss = torch.mean(torch.pow(output - data, 2))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output_text = 'Train Epoch: {current_epoch}/{total_epoch}, Loss: {loss:.8f}'.format(
                current_epoch = num + 1, total_epoch = epoch, loss = loss.item())
            print(output_text)

        output_file = open(log_file_name, 'a')
        output_text = output_text + '\n'
        output_file.write(output_text)
        output_file.close()

        if (num + 1) % 5 == 0:
            parameter_name = './{folder_name}/net_params_{epoch}.pkl'.format(folder_name = model_folder_name, epoch = num + 1)
            torch.save(net.state_dict(), parameter_name)

def sampling_module(input, sampling_matrix):
    input = torch.squeeze(input, dim = 1)
    output = torch.split(input, 33, dim = 1)
    output = torch.cat(output, dim = 0)
    output = torch.split(output, 33, dim = 2)
    output = torch.cat(output, dim = 0)
    output = torch.reshape(output, [-1, 33 * 33])
    output = torch.transpose(output, 0, 1)
    output = torch.matmul(sampling_matrix, output)
    return output

class Denoiser(nn.Module):
    def __init__(self):
        super(Denoiser, self).__init__()
        self.conv1 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv4 = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))
        self.alpha_step = nn.Parameter(torch.Tensor([1.0])).float()

    def forward(self, input, y, sampling_matrix):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        samp_x = torch.matmul(sampling_matrix, input)
        z = y - samp_x
        identity = torch.eye(33 * 33).float()
        identity = identity.to(device)
        h = self.alpha_step * torch.matmul(torch.transpose(self.sampling_matrix, 0, 1), self.sampling_matrix) - identity

        noise = torch.transpose(input, 0, 1)
        noise = torch.reshape(noise, [-1, 33, 33])
        noise = torch.unsqueeze(noise, dim = 1)

        noise = F.conv2d(noise, self.conv1, padding = 1)
        noise = F.relu(noise)
        noise = F.conv2d(noise, self.conv2, padding = 1)
        noise = F.relu(noise)
        noise = F.conv2d(noise, self.conv3, padding = 1)
        noise = F.relu(noise)
        noise = F.conv2d(noise, self.conv4, padding = 1)

        noise = torch.squeeze(noise, dim = 1)
        noise = torch.reshape(noise, [-1, 33 * 33])
        noise = torch.transpose(noise, 0, 1)

        x = input + self.alpha_step * torch.matmul(torch.transpose(self.sampling_matrix, 0, 1), z)
        output = x - torch.matmul(h, noise)

        return output

class Deblocker(nn.Module):
    def __init__(self):
        super(Deblocker, self).__init__()
        self.conv1 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv4 = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, input):
        noise = torch.transpose(input, 0, 1)
        noise = torch.reshape(noise, [-1, 33, 33])
        noise = torch.unsqueeze(noise, dim = 1)

        noise = F.conv2d(noise, self.conv1, padding = 1)
        noise = F.relu(noise)
        noise = F.conv2d(noise, self.conv2, padding = 1)
        noise = F.relu(noise)
        noise = F.conv2d(noise, self.conv3, padding = 1)
        noise = F.relu(noise)
        noise = F.conv2d(noise, self.conv4, padding = 1)

        noise = torch.squeeze(noise, dim = 1)
        noise = torch.reshape(noise, [-1, 33 * 33])
        noise = torch.transpose(noise, 0, 1)

        output = input - noise

        return output

class AMPNet(nn.Module):
    def __init__(self, total_layer, sampling_matrix):
        super(AMPNet, self).__init__()
        self.sampling_matrix = nn.Parameter(sampling_matrix).float()
        self.initial_matrix = nn.Parameter(torch.tranpose(sampling_matrix, 0, 1)).float()
        self.total_layer = total_layer
        self.denoiser = []
        self.deblocker = []

        for phase in range(total_layer):
            self.denoiser.append(Denoiser())
            self.deblocker.append(Deblocker())

        for num, denoiser in enumerate(self.denoiser):
            self.add_module('denoiser_' + str(num + 1), denoiser)

        for num, deblocker in enumerate(self.deblocker):
            self.add_module('deblocker_' + str(num + 1), deblocker)

    def forward(self, input):
        H = int(input.shape[2] / 33)
        L = int(input.shape[3] / 33)
        S = input.shape[0]
        y = sampling_module(input, self.sampling_matrix)
        x = torch.matmul(self.initial_matrix, y)

        for phase in range(self.total_layer):
            denoiser = self.denoiser[phase]
            deblocker = self.deblocker[phase]

            x = denoiser(x, y, self.sampling_matrix)
            x = deblocker(x)

        output = torch.transpose(x, 0, 1)
        output = torch.reshape(output, [-1, 33, 33])
        output = torch.split(output, H * S, dim = 0)
        output = torch.cat(output, dim = 2)
        output = torch.split(output, S, dim = 0)
        output = torch.cat(output, dim = 1)
        return output

if __name__ == '__main__':
    main()