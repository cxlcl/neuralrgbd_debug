'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

import numpy as np
import math

import matplotlib as mlt
mlt.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils.models as utils_model
import mutils.misc as m_misc 

import scipy.io as sio
import mio.imgIO as imgIO

import PIL.Image as image

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

MIN_DEPTH = 1
MAX_DEPTH = 60

def cat_imgs(img_names, output_name ):
    imgs = [np.array( image.open(imgname)) for imgname in img_names]
    imgs = np.hstack(imgs)
    plt.imsave(output_name, imgs)


def depth_regression(Depth_Indx_vol, BV):
    '''
    Depth regression
    '''
    return torch.sum((torch.exp(BV.detach()) * Depth_Indx_vol).squeeze(), dim=0).squeeze().cpu().numpy()

def export_res_img( ref_dat, BV_measure, d_candi, resfldr, batch_idx,
                    depth_scale = 1000, conf_scale = 1000):


    # depth map #
    nDepth = len(d_candi)
    dmap_height, dmap_width = BV_measure.shape[2], BV_measure.shape[3] 
    Depth_val_vol = torch.ones(1, nDepth,  dmap_height, dmap_width).cuda()

    for idepth in range(nDepth):
        Depth_val_vol[0, idepth, ...] = Depth_val_vol[0, idepth, ...] * d_candi[idepth]
    dmap_th = depth_regression(Depth_val_vol, BV_measure)
    dmap = torch.FloatTensor(dmap_th).cpu().numpy()     ## pred_depth


    # confMap #
    confMap_log, _ = torch.max(BV_measure, dim=1)
    confMap_log = torch.exp(confMap_log.squeeze().cpu())
    confMap_log = confMap_log.cpu().numpy()
    confmap = torch.FloatTensor(confMap_log).unsqueeze(0).unsqueeze(0).cuda() 
    confmap = confmap.squeeze().cpu().numpy()
    img = ref_dat['img']
    img = img.squeeze().cpu().permute(1,2,0).numpy()
    img_in_png = _un_normalize( img ); img_in_png = (img_in_png * 255).astype(np.uint8)

    # write to path #
    m_misc.m_makedir(resfldr)
    img_path = '%s/img_%05d.png'%(resfldr, batch_idx)
    d_path = '%s/d_%05d.pgm'%(resfldr, batch_idx)
    conf_path = '%s/conf_%05d.pgm'%(resfldr, batch_idx)
    d_vis_path = '%s/d_vis_%05d.png'%(resfldr, batch_idx)  ### add

    plt.imsave(img_path, img_in_png)
    # plt.imsave(d_vis_path, 1./ dmap, cmap='plasma')  ### add
    imgIO.export2pgm( d_path,    (dmap * depth_scale ).astype(np.uint16) )
    imgIO.export2pgm( conf_path, (confmap * conf_scale ).astype(np.uint16) )

    gt = ref_dat['dmap_imgsize']
    mask = np.logical_and(gt > MIN_DEPTH, gt < MAX_DEPTH)

    ratio = np.median(gt[mask]) / np.median(dmap[mask])
    dmap *= ratio

    dmap[dmap < MIN_DEPTH] = MIN_DEPTH
    dmap[dmap > MAX_DEPTH] = MAX_DEPTH

    return torch.tensor(dmap).unsqueeze(0), ref_dat['dmap_imgsize']


def export_res_refineNet(ref_dat, BV_measure, d_candi,  res_fldr, batch_idx, diff_vrange_ratio=4, 
        cam_pose = None, cam_intrinM = None, output_pngs = False, save_mat=True, output_dmap_ref=True):
    '''
    export results
    ''' 

    # img_in #
    img_up = ref_dat['img']
    img_in_raw = img_up.squeeze().cpu().permute(1,2,0).numpy()
    img_in = (img_in_raw - img_in_raw.min()) / (img_in_raw.max()-img_in_raw.min()) * 255.

    # confMap #
    confMap_log, _ = torch.max(BV_measure, dim=1)
    confMap_log = torch.exp(confMap_log.squeeze().cpu())
    confMap_log = confMap_log.cpu().numpy()

    # depth map #
    nDepth = len(d_candi)
    dmap_height, dmap_width = BV_measure.shape[2], BV_measure.shape[3] 
    dmap = m_misc.depth_val_regression(BV_measure, d_candi, BV_log = True).squeeze().cpu().numpy()  # (256, 768)

    gt = ref_dat['dmap_rawsize']
    raw_w, raw_h = gt.shape[1], gt.shape[2]

    # save up-sampeled results #
    resfldr = res_fldr 
    m_misc.m_makedir(resfldr)

    img_up_path ='%s/input.png'%(resfldr,)
    conf_up_path = '%s/conf.png'%(resfldr,)
    dmap_raw_path = '%s/dmap_raw.png'%(resfldr,)
    final_res_up = '%s/res_%05d.png'%(resfldr, batch_idx) 

    if output_dmap_ref: # output GT depth
        ref_up = '%s/dmap_ref.png'%(resfldr,)
        res_up_diff = '%s/dmaps_diff.png'%(resfldr,)
        dmap_ref = ref_dat['dmap_imgsize']
        dmap_ref = dmap_ref.squeeze().cpu().numpy() 
        mask_dmap = (dmap_ref > 0 ).astype(np.float)
        dmap_diff_raw = np.abs(dmap_ref - dmap ) * mask_dmap
        dmaps_diff = dmap_diff_raw 
        plt.imsave(res_up_diff, dmaps_diff, vmin=0, vmax=d_candi.max()/ diff_vrange_ratio )
        plt.imsave(ref_up, dmap_ref, vmax= d_candi.max(), vmin=0, cmap='gray')

    plt.imsave(conf_up_path, confMap_log, vmin=0, vmax=1, cmap='jet')
    plt.imsave(dmap_raw_path, dmap, vmin=0., vmax =d_candi.max(), cmap='gray' )
    plt.imsave(img_up_path, img_in.astype(np.uint8))

    # output the depth as .mat files # 
    fname_mat = '%s/depth_%05d.mat'%(resfldr, batch_idx)
    img_path = ref_dat['img_path'] 
    if save_mat:
        if not output_dmap_ref:
            mdict = { 'dmap': dmap, 'img': img_in_raw, 'confMap': confMap_log, 'img_path': img_path}
        elif cam_pose is None:
            mdict = {'dmap_ref': dmap_ref, 'dmap': dmap, 'img': img_in_raw, 'confMap': confMap_log,
                    'img_path':   img_path}
        else:
            mdict = {'dmap_ref': dmap_ref, 'dmap': dmap, 
                    'img': img_in_raw, 'cam_pose': cam_pose, 
                    'confMap':confMap_log, 'cam_intrinM': cam_intrinM, 
                    'img_path': img_path } 
        sio.savemat(fname_mat, mdict) 

    # print('export to %s'%(final_res_up))
    
    if output_dmap_ref:
        cat_imgs((img_up_path, conf_up_path, dmap_raw_path, res_up_diff, ref_up), final_res_up) 
    else:
        cat_imgs((img_up_path, conf_up_path, dmap_raw_path), final_res_up) 

    if output_pngs:
        import cv2
        png_fldr = '%s/output_pngs'%(res_fldr, )
        m_misc.m_makedir( png_fldr ) 
        depth_png = (dmap * 1000 ).astype(np.uint16)
        img_in_png = _un_normalize( img_in_raw ); img_in_png = (img_in_png * 255).astype(np.uint8)
        confMap_png = (confMap_log*255).astype(np.uint8) 
        cv2.imwrite( '%s/d_%05d.png'%(png_fldr, batch_idx), depth_png)
        cv2.imwrite( '%s/rgb_%05d.png'%(png_fldr, batch_idx), img_in_png)
        cv2.imwrite( '%s/conf_%05d.png'%(png_fldr, batch_idx), confMap_png)

        if output_dmap_ref:
            depth_ref_png = (dmap_ref * 1000).astype(np.uint16) 
            cv2.imwrite( '%s/dref_%05d.png'%(png_fldr, batch_idx), depth_ref_png)


def do_evaluation(ref_dat, BV_measure, d_candi):
    dmap = m_misc.depth_val_regression(BV_measure, d_candi, BV_log = True).squeeze().cpu().numpy()  # (256, 768)

    gt = ref_dat['dmap_rawsize']
    raw_w, raw_h = gt.shape[1], gt.shape[2]

    dmap = image.fromarray(dmap)
    pred_depth = dmap.resize((raw_h, raw_w), image.NEAREST)
    pred_depth = torch.FloatTensor(np.array(pred_depth)).unsqueeze(0)

    return pred_depth, gt


def _un_normalize( img_in ):
    img_out = np.zeros( img_in.shape )
    for ich in range(3):
        img_out[:, :, ich] = img_in[:, :, ich] * __imagenet_stats['std'][ich]
        img_out[:, :, ich] +=  __imagenet_stats['mean'][ich]

    return img_out

def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x.float()) / math.log(10)

class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0
        self.squarel, self.rmselog = 0, 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.squarel, self.rmselog = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, squarel, rmselog, delta1, delta2, delta3, gpu_time, data_time):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time
        self.squarel, self.rmselog = squarel, rmselog

    def evaluate(self, output, target):
        output = output.float()
        target = target.float()
        valid_mask = target>0
        output = output[valid_mask]
        target = target[valid_mask]

        abs_diff = (output.float() - target.float()).abs()
        abs_diff_log = (log10(output.float()) - log10(target.float())).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())
        self.squarel = float((torch.pow(abs_diff, 2) / target).mean())
        self.rmselog = math.sqrt(float((torch.pow(abs_diff_log, 2)).mean()))

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_squarel, self.sum_rmselog = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0
        self.squarel, self.rmselog = 0, 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n

        self.sum_irmse += n*result.irmse
        self.sum_imae += n*result.imae
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_absrel += n*result.absrel
        self.sum_lg10 += n*result.lg10
        self.sum_delta1 += n*result.delta1
        self.sum_delta2 += n*result.delta2
        self.sum_delta3 += n*result.delta3
        self.sum_data_time += n*data_time
        self.sum_gpu_time += n*gpu_time
        self.sum_squarel += n * result.squarel
        self.sum_rmselog += n * result.rmselog

    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count,
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_squarel / self.count, self.sum_rmselog / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
            self.sum_gpu_time / self.count, self.sum_data_time / self.count)
        return avg
