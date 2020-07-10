import os
import json
import time
import importlib
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import skimage.measure as measure #tcw201904101622tcw
from torch.autograd import Variable
from dataset import TestDataset
from PIL import Image
import cv2 #201904111751tcwi
from torchsummary import summary #tcw20190623
#from torchsummaryX import summary #tcw20190625
os.environ['CUDA_VISIBLE_DEVICES']='0'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--sample_dir", type=str)
    parser.add_argument("--test_data_dir", type=str, default="dataset/Urban100")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--shave", type=int, default=20)

    return parser.parse_args()


def save_image(tensor, filename):
    tensor = tensor.cpu()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

def psnr(im1, im2): #tcw201904101621
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = measure.compare_psnr(im1, im2, data_range=1)
    return psnr
#tcw20190413043
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
def rgb2ycbcr(img, only_y=True):
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

def sample(net, device, dataset, cfg):
    scale = cfg.scale
    mean_psnr = 0
    mean_psnr1 = 0
    mean_psnr2 = 0
    mean_ssim = 0 #tcw20190413047
    for step, (hr, lr, name) in enumerate(dataset):
        if "DIV2K" in dataset.name:
            t1 = time.time()
            h, w = lr.size()[1:]
            h_half, w_half = int(h/2), int(w/2)
            h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

            lr_patch = torch.tensor((4, 3, h_chop, w_chop), dtype=torch.float)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
            lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
            lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
            lr_patch = lr_patch.to(device)
            
            sr = net(lr_patch, cfg.scale).detach()
            
            h, h_half, h_chop = h*scale, h_half*scale, h_chop*scale
            w, w_half, w_chop = w*scale, w_half*scale, w_chop*scale

            result = torch.tensor((3, h, w), dtype=torch.float).to(device)
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
            sr = result
            t2 = time.time()
        else:
            t1 = time.time()
            #print '--------'
            #print lr.size() #e.g (3,512,512)
            lr = lr.unsqueeze(0).to(device)
            #print lr.size() #(1,3,512,512)
            #b = net(lr, cfg.scale).detach()
            #print b.size()  #(1,3,1024,1024)
            sr = net(lr, cfg.scale).detach().squeeze(0) #detach() break the reversed transformation.
            #print sr.size() #(3,1024,1024)
            lr = lr.squeeze(0)
            #print lr.size() #(3,512,512)
            t2 = time.time()
        #print 'step is %d, mean_psnr is %d' % (step,mean_psrn) 
        #print cfg.ckpt_path #./checkpoint/carn.pth
        #print cfg.ckpt_path.split(".") #['', '/checkpoint/carn', 'pth']
        #print cfg.ckpt_path.split(".")[0] # ''
        #print cfg.ckpt_path.split(".")[0].split("/") #''
        #print cfg.ckpt_path.split(".")[0].split("/")[-1] #''
        model_name = cfg.ckpt_path.split(".")[0].split("/")[-1] #''
        #print 
        #print 'a'
        #print '%s', %(model_name)
        #print model_name
        sr_dir = os.path.join(cfg.sample_dir,
                              model_name, 
                              cfg.test_data_dir.split("/")[-1],
                              "x{}".format(cfg.scale),
                              "SR")
        hr_dir = os.path.join(cfg.sample_dir,
                              model_name, 
                              cfg.test_data_dir.split("/")[-1],
                              "x{}".format(cfg.scale),
                              "HR")
        #print sr_dir #sample/Urban100/x2/SR
        #print hr_dir #sample/Urban100/x2/HR
        if not os.path.exists(sr_dir): #201904072208 tcw
            #os.makedirs(sr_dir, exist_ok=Ture)             #201904072211tcw, it is given at first, but it is wrong. So, I mark it.
            #os.makedirs(hr_dir, exist_ok=Ture)
            os.makedirs(sr_dir, mode=0o777)
        if not os.path.exists(hr_dir): #201904072208 tcw
            os.makedirs(hr_dir, mode=0o777)

        sr_im_path = os.path.join(sr_dir, "{}".format(name.replace("HR", "SR"))) #use SR instead of HR in the name of high-resolution image
        hr_im_path = os.path.join(hr_dir, "{}".format(name))#name is a name of high-resolution image
        #print sr_im_path #sample/Urban100/x2/SR/img_100_SRF_2_SR.png
        #print hr_im_path #sample/Urban100/x2/HR/img_100_SRF_2_HR.png
        save_image(sr, sr_im_path)
        save_image(hr, hr_im_path)
        #201904111731tcw y
        #sr_Y = cv2.imread(sr_im_path,cv2.IMREAD_COLOR)
        #sr_Y = cv2.cvtColor(sr_Y,cv2.COLOR_BGR2YCrCb) #YCrCb
        #sr_Y = rgb2ycbcr(sr_Y)
        #sr_Y = sr_Y[:,:,0] #y
        #print sr_Y.shape
        #hr_Y = cv2.imread(hr_im_path,cv2.IMREAD_COLOR)
        #hr_Y = cv2.cvtColor(hr_Y,cv2.COLOR_BGR2YCrCb) #YCrCb
        #hr_Y = rgb2ycbcr(hr_Y)
        #hr_Y = hr_Y[:,:,0] #y 
        #print hr_Y.shape
        hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy() #(644.1024,3) is the same dimensional with the size of input test image in dataset.py. #201904101617tcw
        sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()#tcw201904101617
        bnd = scale  #tcw
        #''''''''''''
        #hr_Y1 = cv2.cvtColor(hr,cv2.COLOR_BGR2YCrCb) #YCrCb
        #sr_Y1 = cv2.cvtColor(sr,cv2.COLOR_BGR2YCrCb) #YCrCb
        #hr_Y2 =hr_Y1[:,:,0]
        #sr_Y2 = sr_Y1[:,:,0]
        #hr_Y2 = hr_Y2[bnd:-bnd, bnd:-bnd] #tcw
        #sr_Y2 = sr_Y2[bnd:-bnd, bnd:-bnd] #tcw
        #''''''''''''
        #im1 = hr[bnd:-bnd, bnd:-bnd] #tcw
        #im2 = sr[bnd:-bnd, bnd:-bnd] #tcw
        sr_1 = rgb2ycbcr(sr)
        hr_1 = rgb2ycbcr(hr)
        #sr_Y = sr_Y[bnd:-bnd,bnd:-bnd]#tcw201904111837
        #hr_Y = hr_Y[bnd:-bnd,bnd:-bnd]#tcw201904111837
        sr_1 = sr_1[bnd:-bnd,bnd:-bnd]#tcw201904111837
        hr_1 = hr_1[bnd:-bnd,bnd:-bnd]#tcw201904111837
        #mean_psnr1 +=psnr(sr_Y,hr_Y)/len(dataset)
        mean_psnr2 +=psnr(sr_1,hr_1)/len(dataset)
        mean_ssim += calculate_ssim(sr_1,hr_1)/len(dataset)
        #print mean_psnr2, mean_ssim, len(dataset)
        #print len(dataset) #it is only used to debug the code.
        #mean_psnr += psnr(im1, im2) / len(dataset) #tcw
        #mean_psnr2 +=psnr(sr_Y2,hr_Y2)/len(dataset)
        #print("Saved {} ({}x{} -> {}x{}, {:.3f}s)"
           # .format(sr_im_path, lr.shape[1], lr.shape[2], sr.shape[1], sr.shape[2], t2-t1))
        #Saved sample/Urban100/x2/SR/img_100_SRF_2_SR.png (512x512 -> 1024x1024, 0.007s)
    #print mean_psnr, mean_psnr1, mean_psnr2 #tcw
    #print mean_psnr1, mean_psnr2
    print (mean_psnr2,  mean_ssim)
        #print '-------------------'z

def main(cfg):
    module = importlib.import_module("model.{}".format(cfg.model))
    ''' 
    net = module.Net(multi_scale=False, 
                     group=cfg.group)
    '''
    net = module.Net(scale=cfg.scale, 
                     group=cfg.group)
    '''
    #net = MyModel
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
	#print('' + str(list(i.size())))
	for j in i.size():
            l *= j
	    #print('' + str(l))
	    k = k + l
	print(''+ str(k))
    '''
    print(json.dumps(vars(cfg), indent=4, sort_keys=True)) #print cfg information according order.
    state_dict = torch.load(cfg.ckpt_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:] # remove "module."
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    #os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #0 is number of gpu, if this gpu1 is work, you can set it into 1 (device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))
    net = net.to(device)
    #summary(net,[(3,240, 160),(3,1000, 2000)]) #tcw20190623
    #summary(net,[torch.zeros(1,3,240,160),2],2)
    dataset = TestDataset(cfg.test_data_dir, cfg.scale)
    sample(net, device, dataset, cfg)
 

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
