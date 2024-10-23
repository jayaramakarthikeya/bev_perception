"""
code adapted from https://github.com/nv-tlabs/lift-splat-shoot
and also https://github.com/wayveai/fiery/blob/master/fiery/data.py
"""

from ast import Dict
from sympy import centroid
import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box

from src.utils import vox_utils 
from src.utils import data_utils
from src.utils import geom_utils
import itertools
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig

discard_invisible = False



class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, dataroot, is_train, data_aug_conf, centroid=None, bounds=None, res_3d=None, nsweeps=1, seqlen=1, refcam_id=1, get_tids=False, temporal_aug=False, use_radar_filters=False, do_shuffle_cams=True):
        self.nusc = nusc
        self.dataroot = dataroot
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        # self.grid_conf = grid_conf
        self.nsweeps = nsweeps
        self.use_radar_filters = use_radar_filters
        self.do_shuffle_cams = do_shuffle_cams
        self.res_3d = res_3d
        self.bounds = bounds
        self.centroid = centroid

        self.seqlen = seqlen
        self.refcam_id = refcam_id

                    

        
        
        self.scenes = self.get_scenes()
        
        # print('applying hack to use just first scene')
        # self.scenes = self.scenes[0:1]
        
        self.ixes = self.prepro()
        if temporal_aug:
            self.indices = self.get_indices_tempaug()
        else:
            self.indices = self.get_indices()

        self.get_tids = get_tids

        # print('ixes', self.ixes.shape)
        print('indices', self.indices.shape)

        if self.bounds is not None and self.res_3d is not None:
            XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = self.bounds
            self.Z, self.Y, self.X = self.res_3d

            grid_conf = { # note the downstream util uses a different XYZ ordering
                'xbound': [XMIN, XMAX, (XMAX-XMIN)/float(self.X)],
                'ybound': [ZMIN, ZMAX, (ZMAX-ZMIN)/float(self.Z)],
                'zbound': [YMIN, YMAX, (YMAX-YMIN)/float(self.Y)],
            }
            dx, bx, nx = data_utils.gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
            self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()
        else:
            print("enter the bounds and res_3d")
            exit()

        self.vox_util = vox_utils.Vox_util(
            self.Z, self.Y, self.X,
            scene_centroid=torch.from_numpy(self.centroid).float().cuda(),
            bounds=self.bounds,
            assert_cube=False)

        print(self)
    
    def get_scenes(self):
        
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]
        scenes = create_splits_scenes()[split]
        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]
        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]
        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))
        return samples
    
    def get_indices(self):
        indices = []
        for index in range(len(self.ixes)):
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.seqlen):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.ixes):
                    is_valid_data = False
                    break
                rec = self.ixes[index_t]
                # Check if scene is the same
                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                index += 38
                indices.append(current_indices)

        return np.asarray(indices)

    def get_indices_tempaug(self):
        indices = []
        t_patterns = None
        if self.seqlen == 1:
            return self.get_indices()
        elif self.seqlen == 2:
            # seq options: (t, t+1), (t, t+2)
            t_patterns = [[0,1], [0,2]]
        elif self.seqlen == 3:
            # seq options: (t, t+1, t+2), (t, t+1, t+3), (t, t+2, t+3)
            t_patterns = [[0,1,2], [0,1,3], [0,2,3]]
        elif self.seqlen == 5:
            t_patterns = [
                    [0,1,2,3,4], # normal
                    [0,1,2,3,5], [0,1,2,4,5], [0,1,3,4,5], [0,2,3,4,5], # 1 skip
                    # [1,0,2,3,4], [0,2,1,3,4], [0,1,3,2,4], [0,1,2,4,3], # 1 reverse
                    ]
        else:
            raise NotImplementedError("timestep not implemented")

        for index in range(len(self.ixes)):
            for t_pattern in t_patterns:
                is_valid_data = True
                previous_rec = None
                current_indices = []
                for t in t_pattern:
                    index_t = index + t
                    # going over the dataset size limit
                    if index_t >= len(self.ixes):
                        is_valid_data = False
                        break
                    rec = self.ixes[index_t]
                    # check if scene is the same
                    if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                        is_valid_data = False
                        break
                    
                    current_indices.append(index_t)
                    previous_rec = rec

                if is_valid_data:
                    indices.append(current_indices)
                    # indices.append(list(reversed(current_indices)))
                    # indices += list(itertools.permutations(current_indices))

        return np.asarray(indices)
    
    def sample_augmentation(self):
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            if 'resize_lim' in self.data_aug_conf and self.data_aug_conf['resize_lim'] is not None:
                resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            else:
                resize = self.data_aug_conf['resize_scale']

            resize_dims = (int(fW*resize), int(fH*resize))

            newW, newH = resize_dims

            # center it
            crop_h = int((newH - fH)/2)
            crop_w = int((newW - fW)/2)

            crop_offset = self.data_aug_conf['crop_offset']
            crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
            crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        else: # validation/test
            # do a perfect resize
            resize_dims = (fW, fH)
            crop_h = 0
            crop_w = 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize_dims, crop

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])

            imgname = os.path.join(self.dataroot, samp['filename'])
            img = Image.open(imgname)
            W, H = img.size

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            resize_dims, crop = self.sample_augmentation()

            sx = resize_dims[0]/float(W)
            sy = resize_dims[1]/float(H)

            intrin = geom_utils.scale_intrinsics(intrin.unsqueeze(0), sx, sy).squeeze(0)

            fx, fy, x0, y0 = geom_utils.split_intrinsics(intrin.unsqueeze(0))

            new_x0 = x0 - crop[0]
            new_y0 = y0 - crop[1]

            pix_T_cam = geom_utils.merge_intrinsics(fx, fy, new_x0, new_y0)
            intrin = pix_T_cam.squeeze(0)

            img = data_utils.img_transform(img, resize_dims, crop)
            imgs.append(data_utils.totorch_img(img))

            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)

            
        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),torch.stack(intrins))


    def get_lidar_data(self, rec, nsweeps):
        pts = data_utils.get_lidar_data(self.nusc, rec, nsweeps=nsweeps, min_distance=2.2, dataroot=self.dataroot)
        return pts

    def get_radar_data(self, rec, nsweeps):
        pts = data_utils.get_radar_data(self.nusc, rec, nsweeps=nsweeps, min_distance=2.2, use_radar_filters=self.use_radar_filters, dataroot=self.dataroot)
        return torch.Tensor(pts)

    def get_binimg(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        for ii, tok in enumerate(rec['anns']):
            inst = self.nusc.get('sample_annotation', tok)
            
            # NuScenes filter
            if 'vehicle' not in inst['category_name']:
                continue
            if discard_invisible and int(inst['visibility_token']) == 1:
                # filter invisible vehicles
                continue
                
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], ii+1)  # type: ignore
        


        return torch.Tensor(img).unsqueeze(0), torch.Tensor(data_utils.convert_egopose_to_matrix_numpy(egopose))

    def get_seg_bev(self, lrtlist_cam, vislist):
        B, N, D = lrtlist_cam.shape
        assert(B==1)

        seg = np.zeros((self.Z, self.X))
        val = np.ones((self.Z, self.X))

        corners_cam = geom_utils.get_xyzlist_from_lrtlist(lrtlist_cam) # B, N, 8, 3
        y_cam = corners_cam[:,:,:,1] # y part; B, N, 8
        corners_mem = self.vox_util.Ref2Mem(corners_cam.reshape(B, N*8, 3), self.Z, self.Y, self.X).reshape(B, N, 8, 3)

        # take the xz part
        corners_mem = torch.stack([corners_mem[:,:,:,0], corners_mem[:,:,:,2]], dim=3) # B, N, 8, 2
        # corners_mem = corners_mem[:,:,:4] # take the bottom four

        for n in range(N):
            _, inds = torch.topk(y_cam[0,n], 4, largest=False)
            pts = corners_mem[0,n,inds].numpy().astype(np.int32) # 4, 2

            # if this messes in some later conditions,
            # the solution is to draw all combos
            pts = np.stack([pts[0],pts[1],pts[3],pts[2]])
            
            # pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(seg, [pts], n+1.0) # type: ignore
            
            if vislist[n]==0:
                # draw a black rectangle if it's invisible
                cv2.fillPoly(val, [pts], 0.0) # type: ignore

        return torch.Tensor(seg).unsqueeze(0), torch.Tensor(val).unsqueeze(0) # 1, Z, X

    def get_center_and_offset_bev(self, lrtlist_cam, seg_bev):
        B, N, D = lrtlist_cam.shape
        assert(B==1)

        lrtlist_mem = self.vox_util.apply_mem_T_ref_to_lrtlist(
            lrtlist_cam, self.Z, self.Y, self.X)
        clist_cam = geom_utils.get_clist_from_lrtlist(lrtlist_cam)
        lenlist, rtlist = geom_utils.split_lrtlist(lrtlist_cam) # B,N,3
        rlist_, tlist_ = geom_utils.split_rt(rtlist.reshape(B*N, 4, 4))

        x_vec = torch.zeros((B*N, 3), dtype=torch.float32, device=rlist_.device)
        x_vec[:, 0] = 1 # 0,0,1 
        x_rot = torch.matmul(rlist_, x_vec.unsqueeze(2)).squeeze(2)

        rylist = torch.atan2(x_rot[:, 0], x_rot[:, 2]).reshape(N)
        rylist = geom_utils.wrap2pi(rylist + np.pi/2.0)

        radius = 3
        center, offset = self.vox_util.xyz2circles_bev(clist_cam, radius, self.Z, self.Y, self.X, already_mem=False, also_offset=True)

        masklist = torch.zeros((1, N, 1, self.Z, 1, self.X), dtype=torch.float32)
        for n in range(N):
            inst = (seg_bev==(n+1)).float() # 1, Z, X
            masklist[0,n,0,:,0] = (inst.squeeze() > 0.01).float()

        size_bev = torch.zeros((1, 3, self.Z, self.X), dtype=torch.float32)
        for n in range(N):
            inst = (seg_bev==(n+1)).float() # 1, Z, X
            inst = inst.reshape(self.Z, self.X) > 0.01
            size_bev[0,:,inst] = lenlist[0,n].unsqueeze(1)

        ry_bev = torch.zeros((1, 1, self.Z, self.X), dtype=torch.float32)
        for n in range(N):
            inst = (seg_bev==(n+1)).float() # 1, Z, X
            inst = inst.reshape(self.Z, self.X) > 0.01
            ry_bev[0,:,inst] = rylist[n]
            
        ycoord_bev = torch.zeros((1, 1, self.Z, self.X), dtype=torch.float32)
        for n in range(N):
            inst = (seg_bev==(n+1)).float() # 1, Z, X
            inst = inst.reshape(self.Z, self.X) > 0.01
            ycoord_bev[0,:,inst] = tlist_[n,1] # y part

        offset = offset * masklist
        offset = torch.sum(offset, dim=1) # B,3,Z,Y,X

        min_offset = torch.min(offset, dim=3)[0] # B,2,Z,X
        max_offset = torch.max(offset, dim=3)[0] # B,2,Z,X
        offset = min_offset + max_offset
        
        center = torch.max(center, dim=1, keepdim=True)[0] # B,1,Z,Y,X
        center = torch.max(center, dim=3)[0] # max along Y; 1,Z,X
        
        return center.squeeze(0), offset.squeeze(0), size_bev.squeeze(0), ry_bev.squeeze(0), ycoord_bev.squeeze(0) # 1,Z,X; 2,Z,X; 3,Z,X; 1,Z,X

    def get_lrtlist(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        lrtlist = []
        boxlist = []
        vislist = []
        tidlist = []
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)

            # NuScenes filter
            if 'vehicle' not in inst['category_name']:
                continue
            if int(inst['visibility_token']) == 1:
                vislist.append(torch.tensor(0.0)) # invisible
            else:
                vislist.append(torch.tensor(1.0)) # visible
                
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            tidlist.append(inst['instance_token'])

            # print('rotation', inst['rotation'])
            r = box.rotation_matrix
            t = box.center
            l = box.wlh
            l = np.stack([l[1],l[0],l[2]])
            lrt = data_utils.merge_lrt(l, data_utils.merge_rt(r,t))
            lrt = torch.Tensor(lrt)
            lrtlist.append(lrt)
            ry, _, _ = Quaternion(inst['rotation']).yaw_pitch_roll
            # print('rx, ry, rz', rx, ry, rz)
            rs = np.stack([ry*0, ry, ry*0])
            box_ = torch.from_numpy(np.stack([t,l,rs])).reshape(9)
            # print('box_', box_)
            boxlist.append(box_)
        if len(lrtlist):
            lrtlist = torch.stack(lrtlist, dim=0)
            boxlist = torch.stack(boxlist, dim=0)
            vislist = torch.stack(vislist, dim=0)
            # tidlist = torch.stack(tidlist, dim=0)
        else:
            lrtlist = torch.zeros((0, 19))
            boxlist = torch.zeros((0, 9))
            vislist = torch.zeros((0))
            # tidlist = torch.zeros((0))
            tidlist = []

        return lrtlist, boxlist, vislist, tidlist

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""
    def __len__(self):
        return len(self.indices)
        # return len(self.ixes)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)

        if self.res_3d is not None:
            Z, Y, X = self.res_3d
        
        self.Z, self.Y, self.X = Z, Y, X

    def get_single_item(self, index, cams, refcam_id=None):
        # print('index %d; cam_id' % index, cam_id)
        rec = self.ixes[index]

        imgs, rots, trans, intrins = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=self.nsweeps)
        binimg, egopose = self.get_binimg(rec)
        
        if refcam_id is None:
            if self.is_train:
                # randomly sample the ref cam
                refcam_id = np.random.randint(1, len(cams))
            else:
                refcam_id = self.refcam_id

        # move the target refcam_id to the zeroth slot
        img_ref = imgs[refcam_id].clone()
        img_0 = imgs[0].clone()
        imgs[0] = img_ref
        imgs[refcam_id] = img_0

        rot_ref = rots[refcam_id].clone()
        rot_0 = rots[0].clone()
        rots[0] = rot_ref
        rots[refcam_id] = rot_0
        
        tran_ref = trans[refcam_id].clone()
        tran_0 = trans[0].clone()
        trans[0] = tran_ref
        trans[refcam_id] = tran_0

        intrin_ref = intrins[refcam_id].clone()
        intrin_0 = intrins[0].clone()
        intrins[0] = intrin_ref
        intrins[refcam_id] = intrin_0
        
        #radar_data = self.get_radar_data(rec, nsweeps=self.nsweeps)

        lidar_extra = lidar_data[3:]
        lidar_data = lidar_data[:3]

        lrtlist_, boxlist_, vislist_, tidlist_ = self.get_lrtlist(rec)
        N_ = lrtlist_.shape[0]

        # import ipdb; ipdb.set_trace()
        if N_ > 0:
            
            velo_T_cam = geom_utils.merge_rt(rots, trans)
            cam_T_velo = geom_utils.safe_inverse(velo_T_cam)

            # note we index 0:1, since we already put refcam into zeroth position
            lrtlist_cam = geom_utils.apply_4x4_to_lrt(cam_T_velo[0:1].repeat(N_, 1, 1), lrtlist_).unsqueeze(0)

            seg_bev, valid_bev = self.get_seg_bev(lrtlist_cam, vislist_)
            
            center_bev, offset_bev, size_bev, ry_bev, ycoord_bev = self.get_center_and_offset_bev(lrtlist_cam, seg_bev)
        else:
            seg_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)
            valid_bev = torch.ones((1, self.Z, self.X), dtype=torch.float32)
            center_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)
            offset_bev = torch.zeros((2, self.Z, self.X), dtype=torch.float32)
            size_bev = torch.zeros((3, self.Z, self.X), dtype=torch.float32)
            ry_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)
            ycoord_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)

        N = 150 # i've seen n as high as 103 before, so 150 is probably safe (max number of objects)
        lrtlist = torch.zeros((N, 19), dtype=torch.float32)
        vislist = torch.zeros((N), dtype=torch.float32)
        scorelist = torch.zeros((N), dtype=torch.float32)
        boxlist = torch.zeros((N,9),dtype=torch.float32)
        lrtlist[:N_] = lrtlist_
        vislist[:N_] = vislist_
        boxlist[:N_] = boxlist_
        scorelist[:N_] = 1

        # lidar is shaped 3,V, where V~=26k 
        times = lidar_extra[2] # V
        inds = times==times[0]
        lidar0_data = lidar_data[:,inds]
        lidar0_extra = lidar_extra[:,inds]

        lidar0_data = np.transpose(lidar0_data)
        lidar0_extra = np.transpose(lidar0_extra)
        lidar_data = np.transpose(lidar_data)
        lidar_extra = np.transpose(lidar_extra)
        V = 30000*self.nsweeps
            
        if lidar_data.shape[0] > V:
            # assert(False) # if this happens, it's probably better to increase V than to subsample as below
            lidar0_data = lidar0_data[:V//self.nsweeps]
            lidar0_extra = lidar0_extra[:V//self.nsweeps]
            lidar_data = lidar_data[:V]
            lidar_extra = lidar_extra[:V]
        elif lidar_data.shape[0] < V:
            lidar0_data = np.pad(lidar0_data,[(0,V//self.nsweeps-lidar0_data.shape[0]),(0,0)],mode='constant')
            lidar0_extra = np.pad(lidar0_extra,[(0,V//self.nsweeps-lidar0_extra.shape[0]),(0,0)],mode='constant')
            lidar_data = np.pad(lidar_data,[(0,V-lidar_data.shape[0]),(0,0)],mode='constant')
            lidar_extra = np.pad(lidar_extra,[(0,V-lidar_extra.shape[0]),(0,0)],mode='constant',constant_values=-1)
        lidar0_data = np.transpose(lidar0_data)
        lidar0_extra = np.transpose(lidar0_extra)
        lidar_data = np.transpose(lidar_data)
        lidar_extra = np.transpose(lidar_extra)

        # radar has <700 points 
        # radar_data = np.transpose(radar_data)
        # V = 700*self.nsweeps
        # if radar_data.shape[0] > V:
        #     print('radar_data', radar_data.shape)
        #     print('max pts', V)
        #     assert(False) # i expect this to never happen
        #     radar_data = radar_data[:V]
        # elif radar_data.shape[0] < V:
        #     radar_data = np.pad(radar_data,[(0,V-radar_data.shape[0]),(0,0)],mode='constant')
        # radar_data = np.transpose(radar_data)

        lidar0_data = torch.from_numpy(lidar0_data).float()
        lidar0_extra = torch.from_numpy(lidar0_extra).float()
        lidar_data = torch.from_numpy(lidar_data).float()
        lidar_extra = torch.from_numpy(lidar_extra).float()
        #radar_data = torch.from_numpy(radar_data).float()

        binimg = (binimg > 0).float()
        seg_bev = (seg_bev > 0).float()

        return imgs, rots, trans, intrins, lidar0_data, lidar0_extra, lidar_data, lidar_extra, boxlist, lrtlist, vislist, tidlist_, scorelist, seg_bev, valid_bev, center_bev, offset_bev, size_bev, ry_bev, ycoord_bev, egopose
    
    def __getitem__(self, index):

        cams = self.choose_cams()
        
        if self.is_train and self.do_shuffle_cams:
            # randomly sample the ref cam
            refcam_id = np.random.randint(1, len(cams))
        else:
            refcam_id = self.refcam_id
        
        all_imgs = []
        all_rots = []
        all_trans = []
        all_intrins = []
        all_lidar0_data = []
        all_lidar0_extra = []
        all_lidar_data = []
        all_lidar_extra = []
        all_lrtlist = []
        all_vislist = []
        all_tidlist = []
        all_scorelist = []
        all_seg_bev = []
        all_valid_bev = []
        all_center_bev = []
        all_offset_bev = []
        all_egopose = []
        all_boxlist = []
        for index_t in self.indices[index]:
            # print('grabbing index %d' % index_t)
            if self.get_tids:
                imgs, rots, trans, intrins, lidar0_data, lidar0_extra, lidar_data, lidar_extra, boxlist, lrtlist, vislist, tidlist, scorelist, seg_bev, valid_bev, center_bev, offset_bev, size_bev, ry_bev, ycoord_bev, egopose = self.get_single_item(index_t, cams, refcam_id=refcam_id)
            else:
                imgs, rots, trans, intrins, lidar0_data, lidar0_extra, lidar_data, lidar_extra, boxlist, lrtlist, vislist, _, scorelist, seg_bev, valid_bev, center_bev, offset_bev, size_bev, ry_bev, ycoord_bev, egopose = self.get_single_item(index_t, cams, refcam_id=refcam_id)

            all_imgs.append(imgs)
            all_rots.append(rots)
            all_trans.append(trans)
            all_intrins.append(intrins)
            all_lidar0_data.append(lidar0_data)
            all_lidar0_extra.append(lidar0_extra)
            all_lidar_data.append(lidar_data)
            all_lidar_extra.append(lidar_extra)
            all_lrtlist.append(lrtlist)
            all_vislist.append(vislist)
            all_tidlist.append(tidlist)
            all_scorelist.append(scorelist)
            all_seg_bev.append(seg_bev)
            all_valid_bev.append(valid_bev)
            all_center_bev.append(center_bev)
            all_offset_bev.append(offset_bev)
            all_egopose.append(egopose)
            all_boxlist.append(boxlist)
        
        all_imgs = torch.stack(all_imgs)
        all_rots = torch.stack(all_rots)
        all_trans = torch.stack(all_trans)
        all_intrins = torch.stack(all_intrins)
        all_lidar0_data = torch.stack(all_lidar0_data)
        all_lidar0_extra = torch.stack(all_lidar0_extra)
        all_lidar_data = torch.stack(all_lidar_data)
        all_lidar_extra = torch.stack(all_lidar_extra)
        all_lrtlist = torch.stack(all_lrtlist)
        all_vislist = torch.stack(all_vislist)
        # all_tidlist = torch.stack(all_tidlist)
        all_scorelist = torch.stack(all_scorelist)
        all_seg_bev = torch.stack(all_seg_bev)
        all_valid_bev = torch.stack(all_valid_bev)
        all_center_bev = torch.stack(all_center_bev)
        all_offset_bev = torch.stack(all_offset_bev)
        all_egopose = torch.stack(all_egopose)
        all_boxlist = torch.stack(all_boxlist)
        
        usable_tidlist = -1*torch.ones_like(all_scorelist).long()
        counter = 0
        for t in range(len(all_tidlist)):
            for i in range(len(all_tidlist[t])):
                if t==0:
                    usable_tidlist[t,i] = counter
                    counter += 1
                else:
                    st = all_tidlist[t][i]
                    if st in all_tidlist[0]:
                        usable_tidlist[t,i] = all_tidlist[0].index(st)
                    else:
                        usable_tidlist[t,i] = counter
                        counter += 1
        all_tidlist = usable_tidlist

        return all_imgs, all_rots, all_trans, all_intrins, all_lidar0_data, all_lidar0_extra, all_lidar_data, all_lidar_extra, all_boxlist, all_lrtlist, all_vislist, all_tidlist, all_scorelist, all_seg_bev, all_valid_bev, all_center_bev, all_offset_bev, all_egopose


def worker_rnd_init(x):
    np.random.seed(13 + x)

@hydra.main(config_path="/data/karthik/bev_perception/configs/datamodule",config_name="nuscenes")
def compile_data(config: DictConfig):

    print('loading nuscenes...')
    nusc = NuScenes(version='v1.0-{}'.format(config.params.version),
                    dataroot=config.params.dataroot,
                    verbose=False)
    print('making parser...')
    centroid = np.array([config.params.scene_centroid_x,
                              config.params.scene_centroid_y,
                              config.params.scene_centroid_z]).reshape([1, 3])
    traindata = VizData(
        nusc,
        dataroot=config.params.dataroot,
        is_train=True,
        data_aug_conf=config.params.data_aug_conf,
        nsweeps=config.params.nsweeps,
        centroid=centroid,
        bounds=tuple(config.params.bounds),
        res_3d=tuple(config.params.res_3d),
        seqlen=config.params.seqlen,
        refcam_id=config.params.refcam_id,
        get_tids= config.params.get_tids,
        temporal_aug= config.params.temporal_aug,
        use_radar_filters=config.params.use_radar_filters,
        do_shuffle_cams= config.params.do_shuffle_cams)
    valdata = VizData(
        nusc,
        dataroot=config.params.dataroot,
        is_train=False,
        data_aug_conf=config.params.data_aug_conf,
        nsweeps=config.params.nsweeps,
        centroid=centroid,
        bounds=tuple(config.params.bounds),
        res_3d=tuple(config.params.res_3d),
        seqlen=config.params.seqlen,
        refcam_id=config.params.refcam_id,
        get_tids=config.params.get_tids,
        temporal_aug=False,
        use_radar_filters=config.params.use_radar_filters,
        do_shuffle_cams=False)

    trainloader = torch.utils.data.DataLoader(
        traindata,
        batch_size=config.params.bsz,
        shuffle=config.params.shuffle,
        num_workers=config.params.nworkers,
        drop_last=True,
        worker_init_fn=worker_rnd_init,
        pin_memory=False)
    valloader = torch.utils.data.DataLoader(
        valdata,
        batch_size=config.params.bsz,
        shuffle=config.params.shuffle,
        num_workers=config.params.nworkers_val,
        drop_last=True,
        pin_memory=False)
    print('data ready')
    # for data in trainloader:
    #     #batch = [x.cuda() for x in data if isinstance(x,torch.Tensor)]
    #     #total_memory = torch.cuda.get_device_properties(0).total_memory
    #     # reserved_memory = torch.cuda.memory_reserved(0)
    #     # allocated_memory = torch.cuda.memory_allocated(0)
    #     # free_memory = reserved_memory - allocated_memory

    #     # #Print memory details
    #     # #print(f"Device: {device}")
    #     # print(f"Total Memory: {total_memory / (1024 ** 3):.2f} GB")
    #     # print(f"Available Memory: {free_memory / (1024 ** 3):.2f} GB")
    #     print(data)
    #     break
    return trainloader, valloader


if __name__ == "__main__":
    compile_data()