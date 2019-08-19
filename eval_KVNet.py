#!/usr/bin/env python
# -*- coding:utf-8 -*-

# eval #
import numpy as np
import torch
import time
torch.backends.cudnn.benchmark=True

import mutils.misc as m_misc
import warping.homography as warp_homo
import models.KVNET as m_kvnet
import utils.models as utils_model
import test_utils.export_res as export_res
import test_utils.test_KVNet as test_KVNet

import matplotlib as mlt
mlt.use('Agg')


def check_datArray_pose(dat_array):
    '''
    Check data array pose/dmap for scan-net.
    If invalid pose then use the previous pose.
    Input: data-array: will be modified via reference.
    Output:
    False: not valid, True: valid
    '''
    if_valid = True
    for dat in dat_array:
        if np.isnan(dat['extM'].min()) or np.isnan(dat['extM'].max()):
            if_valid = False
            break

        elif isinstance(dat['extM'], int):
            if_valid = False
            break

    return if_valid


def main():
    import argparse
    print('Parsing the arguments...')
    parser = argparse.ArgumentParser()

    # exp name #
    parser.add_argument('--exp_name', required =True, type=str,
            help='The name of the experiment. Used to naming the folders')

    # about testing #
    parser.add_argument('--model_path', type=str, required=True, help='The pre-trained model path for KV-net')
    parser.add_argument('--split_file', type=str, default=True, help='The split txt file')
    parser.add_argument('--frame_interv', default=5, type=int, help='frame interval')
    parser.add_argument('--t_win', type=int, default = 2, help='The radius of the temporal window; default=2')
    parser.add_argument('--d_min', type=float, default=0, help='The minimal depth value; default=0')
    parser.add_argument('--d_max', type=float, default=5, help='The maximal depth value; default=15')
    parser.add_argument('--ndepth', type=int, default= 64, help='The # of candidate depth values; default= 128')
    parser.add_argument('--sigma_soft_max', type=float, default=10., help='sigma_soft_max, default = 500.')
    parser.add_argument('--feature_dim', type=int, default=64, help='The feature dimension for the feature extractor; default=64')

    # about dataset #
    parser.add_argument('--dataset', type=str, default='scanNet', help='Dataset name: {scanNet, 7scenes, kitti}')
    parser.add_argument('--dataset_path', type=str, default='.', help='Path to the dataset')
    parser.add_argument('--change_aspect_ratio', type=bool, default=False,
                        help='If we want to change the aspect ratio. This option is only useful for KITTI')

    # parsing parameters #
    args = parser.parse_args()
    exp_name = args.exp_name
    dataset_name = args.dataset
    t_win_r = args.t_win
    nDepth = args.ndepth
    d_candi = np.linspace(args.d_min, args.d_max, nDepth)
    sigma_soft_max = args.sigma_soft_max #10.#500.
    dnet_feature_dim = args.feature_dim
    frame_interv = args.frame_interv # should be multiple of 5 for scanNet dataset
    d_upsample = None
    d_candi_dmap_ref = d_candi
    nDepth_dmap_ref = nDepth
    split_file = args.split_file


    # ===== Dataset selection ======== #
    dataset_path = args.dataset_path


    if dataset_name == 'kitti':
        import mdataloader.kitti as dl_kitti
        dataset_init = dl_kitti.KITTI_dataset
        fun_get_paths = lambda traj_indx: dl_kitti.get_paths(traj_indx, split_txt= split_file, mode='val')
        if not dataset_path == '.':
            fun_get_paths = lambda traj_indx: dl_kitti.get_paths(traj_indx, split_txt= split_file,
                    mode='val', database_path_base = dataset_path)
        else: # use default database path
            fun_get_paths = lambda traj_indx: dl_kitti.get_paths(traj_indx, split_txt= split_file,  mode='val')

        if not args.change_aspect_ratio: # we will keep the aspect ratio and do cropping
            img_size = [768, 356]
            crop_w = None
        else: # we will change the aspect ratio and NOT do cropping
            img_size = [768, 356]
            crop_w = None

        n_scenes, _, _, _, _ = fun_get_paths(0)
        traj_Indx = np.arange(0, n_scenes)

    elif dataset_name == 'dm':
        import mdataloader.dm as dl_dm
        dataset_init = dl_dm.DMdataset
        split_file = './mdataloader/dm_split/dm_split.txt' if args.split_file=='.' else args.split_file
        fun_get_paths = lambda traj_indx: dl_dm.get_paths(traj_indx, split_txt= split_file, mode='val')
        if not dataset_path == '.':
            fun_get_paths = lambda traj_indx: dl_dm.get_paths(traj_indx, split_txt= split_file,
                    mode='val', database_path_base = dataset_path)
        else: # use default database path
            fun_get_paths = lambda traj_indx: dl_dm.get_paths(traj_indx, split_txt= split_file,  mode='val')

        if not args.change_aspect_ratio: # we will keep the aspect ratio and do cropping
            img_size = [786, 256]
            crop_w = None
        else: # we will change the aspect ratio and NOT do cropping
            img_size = [786, 256]
            crop_w = None

        n_scenes, _, _, _, _ = fun_get_paths(0)
        traj_Indx = np.arange(0, n_scenes)

    else:
        raise Exception('dataset loader not implemented')

    fldr_path, img_paths, dmap_paths, poses, intrin_path = fun_get_paths(0)
    if dataset_name == 'kitti':
        dataset = dataset_init(True, img_paths, dmap_paths, poses,
                               intrin_path = intrin_path, img_size= img_size, digitize= True,
                               d_candi= d_candi_dmap_ref, resize_dmap=.25, crop_w = crop_w)

        dataset_imgsize = dataset_init(True, img_paths, dmap_paths, poses,
                               intrin_path = intrin_path, img_size= img_size, digitize= True,
                               d_candi= d_candi_dmap_ref, resize_dmap=1)
    else:
        dataset = dataset_init(True, img_paths, dmap_paths, poses,
                               intrin_path = intrin_path, img_size= img_size, digitize= True,
                               d_candi= d_candi_dmap_ref, resize_dmap=.25)

        dataset_imgsize = dataset_init(True, img_paths, dmap_paths, poses,
                               intrin_path = intrin_path, img_size= img_size, digitize= True,
                               d_candi= d_candi_dmap_ref, resize_dmap=1)
    # ================================ #
    print('Initnializing the KV-Net')
    model_KVnet = m_kvnet.KVNET(feature_dim = dnet_feature_dim, cam_intrinsics = dataset.cam_intrinsics,
                                d_candi = d_candi, sigma_soft_max = sigma_soft_max, KVNet_feature_dim = dnet_feature_dim,
                                d_upsample_ratio_KV_net = d_upsample, t_win_r = t_win_r, if_refined = True)

    model_KVnet = torch.nn.DataParallel(model_KVnet)
    model_KVnet.cuda()

    model_path_KV = args.model_path
    print('loading KV_net at %s'%(model_path_KV))
    utils_model.load_pretrained_model(model_KVnet, model_path_KV)
    print('Done')

    rmse, absrel, lg10, squarel, rmselog, D1, D2, D3 = 0, 0, 0, 0, 0, 0, 0, 0

    for traj_idx in traj_Indx:
        res_fldr = '../results/%s/traj_%d'%(exp_name, traj_idx)
        m_misc.m_makedir(res_fldr)
        scene_path_info = []

        print('Getting the paths for traj_%d'%(traj_idx))
        fldr_path, img_seq_paths, dmap_seq_paths, poses, intrin_path =  fun_get_paths(traj_idx)
        dataset.set_paths(img_seq_paths, dmap_seq_paths, poses)

        if dataset_name is 'scanNet':
            # For each trajector in the dataset, we will update the intrinsic matrix #
            dataset.get_cam_intrinsics(intrin_path)

        print('Done')
        dat_array = [ dataset[idx] for idx in range(t_win_r * 2 + 1) ]
        DMaps_meas = []
        traj_length = len(dataset)
        print('trajectory length = %d'%(traj_length))

        average_meter = export_res.AverageMeter()

        ### inference time
        torch.cuda.synchronize()
        start = time.time()

        for frame_cnt, ref_indx in enumerate( range(t_win_r, traj_length - t_win_r - 1) ):
            result = export_res.Result()

            torch.cuda.synchronize()
            data_time = time.time() - start

            eff_iter = True
            valid_seq = check_datArray_pose(dat_array)

            # Read ref. and src. data in the local time window #
            ref_dat, src_dats = m_misc.split_frame_list(dat_array, t_win_r)

            if frame_cnt == 0:
                BVs_predict = None

            if valid_seq and eff_iter:
                # Get poses #
                src_cam_extMs = m_misc.get_entries_list_dict(src_dats, 'extM')
                src_cam_poses = \
                        [warp_homo.get_rel_extrinsicM(ref_dat['extM'], src_cam_extM_) \
                        for src_cam_extM_ in src_cam_extMs ]

                src_cam_poses = [
                        torch.from_numpy(pose.astype(np.float32)).cuda().unsqueeze(0)
                        for pose in src_cam_poses]

                # src_cam_poses size: N V 4 4 #
                src_cam_poses = torch.cat(src_cam_poses, dim=0).unsqueeze(0)
                src_frames = [m_misc.get_entries_list_dict(src_dats, 'img')]

                if frame_cnt == 0 or BVs_predict is None: # the first window for the traj.
                    BVs_predict_in = None
                else:
                    BVs_predict_in = BVs_predict

                # print('testing on %d/%d frame in traj %d/%d ... '%\
                #        (frame_cnt+1, traj_length - 2*t_win_r, traj_idx+1, len(traj_Indx)) )

                torch.cuda.synchronize()
                gpu_time = time.time() - start

                # set trace for specific frame #
                BVs_measure, BVs_predict = test_KVNet.test( model_KVnet, d_candi,
                                                            Ref_Dats = [ref_dat],
                                                            Src_Dats = [src_dats],
                                                            Cam_Intrinsics=[dataset.cam_intrinsics],
                                                            t_win_r = t_win_r,
                                                            Src_CamPoses= src_cam_poses,
                                                            BV_predict= BVs_predict_in,
                                                            R_net = True,
                                                            Cam_Intrinsics_imgsize = dataset_imgsize.cam_intrinsics,
                                                            ref_indx = ref_indx )


                pred_depth, gt = export_res.do_evaluation(ref_dat,  BVs_measure, d_candi_dmap_ref)

                # print(pred_depth.shape, gt.shape)

                result.evaluate(pred_depth.data, gt.data)

                average_meter.update(result, gpu_time, data_time, (traj_length - 2*t_win_r))

                scene_path_info.append( [frame_cnt, dataset[ref_indx]['img_path']] )

            elif valid_seq is False: # if the sequence contains invalid pose estimation
                BVs_predict = None
                print('frame_cnt :%d, include invalid poses'%(frame_cnt ))

            elif eff_iter is False:
                BVs_predict = None

            # Update dat_array #
            dat_array.pop(0)
            dat_array.append(dataset[ref_indx + t_win_r +1 ])

        avg = average_meter.average()

        print('\n*\n'
                  'RMSE={average.rmse:.3f}\n'
                  'AbsRel={average.absrel:.3f}\n'
                  'Log10={average.lg10:.3f}\n'
                  'SquaRel={average.squarel:.3f}\n'
                  'rmselog={average.rmselog:.3f}\n'
                  'Delta1={average.delta1:.3f}\n'
                  'Delta2={average.delta2:.3f}\n'
                  'Delta3={average.delta3:.3f}\n'
                  't_GPU={time:.3f}\n'.format(
                average=avg, time=avg.gpu_time))

            ### inference time
        torch.cuda.synchronize()
        end = time.time()

        rmse += avg.rmse
        absrel += avg.absrel
        lg10 += avg.lg10
        squarel += avg.squarel
        rmselog += avg.rmselog
        D1 += avg.delta1
        D2 += avg.delta2
        D3 += avg.delta3

        print('rmse={%.3f}\n'% (rmse / (traj_idx+1)),
              'absrel={%.3f}\n' % (absrel / (traj_idx+1)),
              'lg10={%.3f}\n' % (lg10 / (traj_idx+1)),
              'squarel={%.3f}\n' % (squarel / (traj_idx+1)),
              'rmselog={%.3f}\n' % (rmselog / (traj_idx+1)),
              'D1={%.3f}\n' % (D1 / (traj_idx+1)),
              'D2={%.3f}\n' % (D2 / (traj_idx+1)),
              'D3={%.3f}\n' % (D3 / (traj_idx+1)))

        print( (end-start)/(traj_length - 2*t_win_r))

        m_misc.save_ScenePathInfo( '%s/scene_path_info.txt'%(res_fldr), scene_path_info )


if __name__ == '__main__':
    main()
