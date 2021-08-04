import os
import random
import numpy as np
import scipy.misc as misc
import skimage.metrics as metrics

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestDataset
import cv2 #20190413029tcw
class Solver():
    def __init__(self, model, cfg):
        if cfg.scale > 0: #there is only a scale, 201904081901
            self.refiner = model(scale=cfg.scale, 
                                 group=cfg.group)
        else: #there is mutile scales,201904081901
            self.refiner = model(multi_scale=True, 
                                 group=cfg.group)
        
        if cfg.loss_fn in ["MSE"]: 
            self.loss_fn = nn.MSELoss()
        elif cfg.loss_fn in ["L1"]: 
            self.loss_fn = nn.L1Loss()
        elif cfg.loss_fn in ["SmoothL1"]:
            self.loss_fn = nn.SmoothL1Loss()

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.refiner.parameters()), 
            cfg.lr)
        
        self.train_data = TrainDataset(cfg.train_data_path, 
                                       scale=cfg.scale, 
                                       size=cfg.patch_size)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=cfg.batch_size,
                                       num_workers=0,
                                       shuffle=True, drop_last=True)
        
        #the ways of chosen GPU
        #the first way
        os.environ['CUDA_VISIBLE_DEVICES']='0,1'
        self.device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu") #tcw201904100941, cuda:1 denotes the GPU of number 1. cuda:0 denotes the GPU of number 0.
        #automically choose the GPU, if torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #"The second way is as follows--------------------------"
        #self.device = torch.device('cuda',1) #the commod is added by tcw 201904100942
        #If torch.device('cuda',1), which chooses the GPU of number 1. If torch.device('cuda',0), which chooses the GPU of number 0.
        self.refiner = self.refiner.to(self.device) #load the model into the self.device
        self.loss_fn = self.loss_fn

        self.cfg = cfg
        self.step = 0
        
        self.writer = SummaryWriter(log_dir=os.path.join("runs", cfg.ckpt_name)) #log
        if cfg.verbose:
            num_params = 0
            for param in self.refiner.parameters():#model.parameters keep the parameters from all the layers.
                num_params += param.nelement() #.nelement() can count the number of all the parameters.
            print("# of params:", num_params)
        
        if not os.path.exists(cfg.ckpt_dir): #201904072208 tcw
            #os.makedirs(cfg.ckpt_dir, exist_ok=True) #201904072211tcw, it is given at first, but it is wrong. So, I mark it.
            os.makedirs(cfg.ckpt_dir, mode=0o777) #2019072211tcw

    def fit(self):
        cfg = self.cfg
        refiner = nn.DataParallel(self.refiner, 
                                  device_ids=range(cfg.num_gpu))
        
        learning_rate = cfg.lr
        while True:
            for inputs in self.train_loader:
                self.refiner.train()
                if cfg.scale > 0: #There is only a scale in the training processing. 
                    scale = cfg.scale
                    hr, lr = inputs[-1][0], inputs[-1][1] #
                else:
                    # only use one of multi-scale data
                    # i know this is stupid but just temporary. it is noticeable that scale is rand.
                    scale = random.randint(2, 4)
                    hr, lr = inputs[scale-2][0], inputs[scale-2][1] #obatin hr,lr under differet scales.
                hr = hr.to(self.device) #load hr on the self.device
                lr = lr.to(self.device)
                
                sr = refiner(lr, scale)
                loss = self.loss_fn(sr, hr)
                
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.refiner.parameters(), cfg.clip) #tcw it is drop out, which can prevent overfitting.
                self.optim.step()

                learning_rate = self.decay_learning_rate()
                for param_group in self.optim.param_groups:
                    param_group["lr"] = learning_rate
                
                self.step += 1
                if cfg.verbose and self.step % cfg.print_interval == 0:
                    if cfg.scale > 0:
                        psnr = self.evaluate("dataset/Urban100", scale=cfg.scale, num_step=self.step)
                        #print 'sdffffffffffff232'
                        self.writer.add_scalar("Urban100", psnr, self.step) #save the data in the file of writer, which is shown via visual figures.
                        #The first parameter is figure name, the second parameter is axis Y, the third parameter is axis X.  
                    else:    
                        psnr = [self.evaluate("dataset/Urban100", scale=i, num_step=self.step) for i in range(2, 5)]
                        self.writer.add_scalar("Urban100_2x", psnr[0], self.step)
                        self.writer.add_scalar("Urban100_3x", psnr[1], self.step)
                        self.writer.add_scalar("Urban100_4x", psnr[2], self.step)
                            
                    self.save(cfg.ckpt_dir, cfg.ckpt_name)

            if self.step > cfg.max_steps: break

    def evaluate(self, test_data_dir, scale=4, num_step=0):
        global mean_psnr, mean_psnr1, mean_ssim
        mean_psnr = 0
        mean_psnr1 = 0
        mean_ssim = 0
        #print 'sdfs'
        cfg = self.cfg
        self.refiner.eval()
        test_data   = TestDataset(test_data_dir, scale=scale)
        test_loader = DataLoader(test_data,
                                 batch_size=1,
                                 num_workers=0,
                                 shuffle=False)

        for step, inputs in enumerate(test_loader): #step is the number of test images
            hr = inputs[0].squeeze(0) #reduce the first dimension
            #inputs[0].size() (1,3,644,1024)
            #inputs[1].size() (1,3,322,512)
            #hr.size() (3,644,1024)
            #lr.size() (3,322,512)
            #lr.size()[1:] ()
            #print inputs[0].size()
            #print inputs[1].size()
            #print hr.size()
            lr = inputs[1].squeeze(0)
            name = inputs[2][0]
            #print lr.size()
            h, w = lr.size()[1:]
            #print lr.size()[1:]
            h_half, w_half = int(h/2), int(w/2)
            h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

            # split large image to 4 patch to avoid OOM error
            lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
            lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
            lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
            lr_patch = lr_patch.to(self.device)
            
            # run refine process in here!
            sr = self.refiner(lr_patch, scale).data
            
            h, h_half, h_chop = h*scale, h_half*scale, h_chop*scale
            w, w_half, w_chop = w*scale, w_half*scale, w_chop*scale
            
            # merge splited patch images
            result = torch.FloatTensor(3, h, w).to(self.device)
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
            sr = result
            hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy() #(644.1024,3) is the same dimensional with the size of input test image in dataset.py.
            sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            #print 'numpy is %d', (hr.shape), which is only used to test, and makes me more easy to understand.
            # evaluate PSNR
            # this evaluation is different to MATLAB version
            # we evaluate PSNR in RGB channel not Y in YCbCR
            #hr1 = hr
            #sr1 = sr
            hr = rgb2ycbcr(hr) #tcw201904122350
            sr = rgb2ycbcr(sr)  
            bnd = scale
            im1 = hr[bnd:-bnd, bnd:-bnd]
            im2 = sr[bnd:-bnd, bnd:-bnd]
            #im3 = hr1[bnd:-bnd, bnd:-bnd]
            #im4 = sr1[bnd:-bnd, bnd:-bnd]
            mean_psnr += psnr(im1, im2) / len(test_data)
            mean_ssim += calculate_ssim(im1,im2)/len(test_data)
            #mean_psnr1 += psnr(im3, im4) / len(test_data)
            #print 'step is %d, mean_psnr is %f' %(step,mean_psnr)
        print("Epoch: {}, M-PSNR: {}, M-SSIM: {}".format(self.step, mean_psnr, mean_ssim))
        #print 'epochs is %d, mean_psnr is %f' %(self.step,mean_psnr1)
        #print mean_psnr#, mean_psnr1
        return mean_psnr
    def load(self, path):
        self.refiner.load_state_dict(torch.load(path))
        '''
        print path
        print path.split(".")
        print path.split(".")[0]
        print path.split(".")[0].split("_")
        print path.split(".")[0].split("_")[-1]
        '''
        splited = path.split(".")[0].split("_")[-1]
        try:
            self.step = int(path.split(".")[0].split("_")[-1])
        except ValueError:
            self.step = 0
        print("Load pretrained {} model".format(path))

    def save(self, ckpt_dir, ckpt_name):
        #print 'sfdfsdsfsdf'
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, self.step))
        torch.save(self.refiner.state_dict(), save_path)

    def decay_learning_rate(self):
        lr = self.cfg.lr * (0.5 ** (self.step // self.cfg.decay))
        return lr

def rgb2ycbcr(img, only_y=True):  #201904122348tcw
    '''
    same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def psnr(im1, im2):
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = metrics.peak_signal_noise_ratio(im1, im2, data_range=1)
    return psnr
#tcw20190413022tcw
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
