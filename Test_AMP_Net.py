import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader

import numpy as np
import scipy.io as sio
import os
import cv2
import glob
from time import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

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
    result_dir = 'result'

    n_input = round(1089 * cs_ratio / 100)
    n_output = 1089
    n_train = 88912
    batch_size = 64
    
    test_name = 'Set11'
    sampling_folder_name = './{folder_name}'.format(folder_name = matrix_dir)
    sampling_matrix_name = './{folder_name}/phi_0_{ratio}_1089_random.mat'.format(folder_name = matrix_dir, ratio = cs_ratio)
    log_folder_name = './{folder_name}'.format(folder_name = log_dir)
    log_file_name = './{folder_name}/PSNR_SSIM_Results_AMP_Net_layer_{layer}_group_{group}_ratio_{ratio}_lr_{lr:.4f}.txt'.format(
        folder_name = log_dir, layer = total_layer, group = group_num, ratio = cs_ratio, lr = learning_rate)
    model_folder_name = './{folder_name}/AMP_Net_layer_{layer}_group_{group}_ratio_{ratio}_lr_{lr:.4f}'.format(
        folder_name = model_dir, layer = total_layer, group = group_num, ratio = cs_ratio, lr = learning_rate)
    test_dir = os.path.join(data_dir, test_name)
    test_file_path = glob.glob(test_dir + '/*.tif')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sampling_data = sio.loadmat(sampling_matrix_name)
    sampling_matrix_np = sampling_data['sampling_matrix']
    sampling_matrix = torch.from_numpy(sampling_matrix_np)
    sampling_matrix = sampling_matrix.to(device)

    net = AMPNet(total_layer, sampling_matrix)
    net = nn.DataParallel(net)
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr = learning_rate)
    parameter_name = './{folder_name}/net_params_{epoch}.pkl'.format(folder_name = model_folder_name, epoch = epoch)
    net.load_state_dict(torch.load(parameter_name))

    image_num = len(test_file_path)
    PSNR_ALL = np.zeros([1, image_num], dtype = np.float32)
    SSIM_ALL = np.zeros([1, image_num], dtype = np.float32)

    net.eval()
    with torch.no_grad():
        for num in range(image_num):
            image_name = test_file_path[num]
            image = cv2.imread(image_name, 1)
            image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            image_rec_yuv = image_yuv.copy()
            image_origin_y = image_yuv[:, :, 0]

            [image_origin, row, col, image_padding, row_new, col_new] = imread_CS_py(image_origin_y)
            image_col = img2col_py(image_padding, 33) / 255.0
            image_padding = image_padding / 255.0

            image_output = image_padding
            H = int(image_padding.shape[1] / 33)
            S = 1

            start = time()

            batch_x = torch.from_numpy(image_output)
            batch_x = torch.unsqueeze(batch_x, dim = 0)
            batch_x = torch.unsqueeze(batch_x, dim = 0)
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x = batch_x.to(device)

            x_output = net(batch_x)

            end = time()

            x_output = torch.reshape(x_output, [-1, 33, 33])
            x_output = torch.split(x_output, H * S, dim = 0)
            x_output = torch.cat(x_output, dim = 2)
            x_output = torch.split(x_output, S, dim = 0)
            x_output = torch.cat(x_output, dim = 1)
            x_output = torch.squeeze(x_output, dim = 0)

            prediction_value = x_output.cpu().data.numpy()
            x_rec = prediction_value[0: row, 0: col] * 255
            rec_PSNR = psnr(x_rec, image_origin.astype(np.float64), data_range = 255)
            rec_SSIM = ssim(x_rec, image_origin.astype(np.float64), data_range = 255)

            print('[{current_img}/{total_img}] Run time for {img_name} is {run_time:.4f}, PSNR is {PSNR:.2f}, SSIM is {SSIM:.4f}'.format(
                current_img = num, total_img = image_num, img_name = image_name, run_time = end - start, PSNR = rec_PSNR, SSIM = rec_SSIM))

            image_rec_yuv[:, :, 0] =x_rec * 255
            image_rec_rgb = cv2.cvtColor(image_rec_yuv, cv2.COLOR_YCrCb2BGR)
            image_rec_rgb = np.clip(image_rec_rgb, 0, 255).astype(np.uint8)

            result_name = image_name.replace(data_dir, result_dir)
            cv2.imwrite('{img_name}_ISTA_Net_ratio_{ratio}_epoch_{epoch}_PSNR_{PSNR:.2f}_SSIM_{SSIM:.4f}.png'.format(
                img_name = result_name, ratio = cs_ratio, epoch = epoch, PSNR = rec_PSNR, SSIM = rec_SSIM), image_rec_rgb)
            del x_output

            PSNR_ALL[0, num] = rec_PSNR
            SSIM_ALL[0, num] = rec_SSIM

    print('\n')
    output_data = 'CS ratio is {ratio}, Average PSNR/SSIM for {test_name} is {PSNR:.2f}/{SSIM:.4f}, Epoch number of model is {epoch} \n'.format(
        ratio = cs_ratio, test_name = test_name, PSNR = np.mean(PSNR_ALL), SSIM = np.mean(SSIM_ALL), epoch = epoch)
    print(output_data)

    log_file = open(log_file_name, 'a')
    log_file.write(output_data)
    log_file.close()

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

class Initializer(nn.Module):
    def __init__(self, sampling_matrix):
        super(Initializer, self).__init__()
        self.initial = nn.Parameter(torch.transpose(sampling_matrix, 0, 1))

    def forward(self, input):
        output = torch.matmul(self.initial, input)
        return [output, self.initial]

class Denoiser(nn.Module):
    def __init__(self, sampling_matrix):
        super(Denoiser, self).__init__()
        self.sampling_matrix = nn.Parameter(sampling_matrix).float()
        self.conv1 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv4 = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))
        self.alpha_step = nn.Parameter(torch.Tensor([1.0])).float()

    def forward(self, input, y, input_origin, initial_matrix):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        H = int(input_origin.shape[2] / 33)
        L = int(input_origin.shape[3] / 33)
        S = input_origin.shape[0]

        samp = torch.matmul(self.sampling_matrix, input)
        z = y - samp
        identity = torch.eye(33 * 33).float()
        identity = identity.to(device)
        h = self.alpha_step * torch.matmul(torch.transpose(self.sampling_matrix, 0, 1), self.sampling_matrix) - identity

        noise_input = torch.transpose(input, 0, 1)
        noise_input = torch.reshape(noise_input, [-1, 33, 33])
        noise_input = torch.unsqueeze(noise_input, dim = 1)

        noise = F.conv2d(noise_input, self.conv1, padding = 1)
        noise = F.relu(noise)
        noise = F.conv2d(noise, self.conv2, padding = 1)
        noise = F.relu(noise)
        noise = F.conv2d(noise, self.conv3, padding = 1)
        noise = F.relu(noise)
        noise = F.conv2d(noise, self.conv4, padding = 1)

        noise = torch.squeeze(noise, dim = 1)
        noise = torch.reshape(noise, [-1, 33 * 33])
        noise = torch.transpose(noise, 0, 1)

        x = torch.matmul(initial_matrix, y)
        x = x + self.alpha_step * torch.matmul(torch.transpose(self.sampling_matrix, 0, 1), z)
        x = x - torch.matmul(h, noise)

        output = torch.reshape(x, [-1, 33, 33])
        output = torch.split(output, H * S, dim = 0)
        output = torch.cat(output, dim = 2)
        output = torch.split(output, S, dim = 0)
        output = torch.cat(output, dim = 1)

        return output

class Deblocker(nn.Module):
    def __init__(self):
        super(Deblocker, self).__init__()
        self.conv1 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv4 = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, input):
        noise_input = torch.transpose(input, 0, 1)
        noise_input = torch.reshape(noise_input, [-1, 33, 33])
        noise_input = torch.unsqueeze(noise_input, dim = 1)

        noise = F.conv2d(noise_input, self.conv1, padding = 1)
        noise = F.relu(noise)
        noise = F.conv2d(noise, self.conv2, padding = 1)
        noise = F.relu(noise)
        noise = F.conv2d(noise, self.conv3, padding = 1)
        noise = F.relu(noise)
        noise = F.conv2d(noise, self.conv4, padding = 1)

        output = noise_input - noise

        return output

class AMPNet(nn.Module):
    def __init__(self, total_layer, sampling_matrix):
        super(AMPNet, self).__init__()
        self.sampling_matrix = sampling_matrix
        self.total_layer = total_layer
        self.initializer = []
        self.denoiser = []
        self.deblocker = []

        for phase in range(total_layer):
            self.initializer.append(Initializer(sampling_matrix))
            self.denoiser.append(Denoiser(sampling_matrix))
            self.deblocker.append(Deblocker())

        for num, initializer in enumerate(self.initializer):
            self.add_module('initializer_'+str(num + 1), initializer)

        for num, denoiser in enumerate(self.denoiser):
            self.add_module('denoiser_'+str(num + 1), denoiser)

        for num, deblocker in enumerate(self.deblocker):
            self.add_module('deblocker_'+str(num + 1), deblocker)

    def forward(self, input):
        y = sampling_module(input, self.sampling_matrix)

        for phase in range(self.total_layer):
            initializer = self.initializer[phase]
            denoiser = self.denoiser[phase]
            deblocker = self.deblocker[phase]

            [output, initial_matrix] = initializer(y)
            output = denoiser(output, y, input, initial_matrix)
            output = deblocker(output)

        return output

def imread_CS_py(image_origin):
    block_size = 33
    [row, col] = image_origin.shape
    row_pad = block_size - np.mod(row, block_size)
    col_pad = block_size - np.mod(col, block_size)
    image_padding = np.concatenate((image_origin, np.zeros([row, col_pad])), axis = 1)
    image_padding = np.concatenate((image_padding, np.zeros([row_pad, col + col_pad])), axis = 0)
    [row_new, col_new] = image_padding.shape
    return [image_origin, row, col, image_padding, row_new, col_new]

def img2col_py(image_padding, block_size):
    [row, col] = image_padding.shape
    row_block = row / block_size
    col_block = col / block_size
    block_num = int(row_block * col_block)
    image_col = np.zeros([block_size ** 2, block_num])
    count = 0
    for x in range(0, row - block_size + 1, block_size):
        for y in range(0, col - block_size + 1, block_size):
            image_col[:, count] = image_padding[x: x + block_size, y: y + block_size].reshape([-1])
            count = count + 1
    return image_col

def col2img_py(x_col, row, col, row_new, col_new):
    block_size = 33
    x0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new - block_size + 1, block_size):
        for y in range(0, col_new - block_size + 1, block_size):
            x0_rec[x: x + block_size, y: y + block_size] = x_col[:, count].reshape([block_size, block_size])
            count = count + 1
    x_rec = x0_rec[: row, : col]
    return x_rec

if __name__ == '__main__':
    main()