#!/usr/bin/env python3
# coding: utf-8

__author__ = 'cleardusk'

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""
import os
import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors, draw_wireframe
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn

from os import listdir
from os.path import join, basename

from DGPT.Utils.Preprocess import RGB2Y
from DGPT.Utils.CUDAFuncs.GaussianBlur import GaussianBlur_CUDA

from utils.trianglemap import matched, simple_tri

STD_SIZE = 120

end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1
def plot_kpt(image, kpt):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    '''
    image = image.copy()
    # kpt = np.round(kpt).astype(np.int32)
    kpt = np.array(kpt)
    # print(kpt.shape, type(kpt))
    # kpt = np.squeeze(kpt, axis=0)
    # kpt = np.transpose(kpt, (2, 1))

    # print("KPT shape", kpt.shape)
    for i in range(kpt.shape[0]):
        pt = kpt[i]
        pt = np.transpose(pt, (1, 0))
        for j in range(pt.shape[0]):
            st = pt[j, :2]
            image = cv2.circle(image,(st[0], st[1]), 1, (0,0,255), 10)
            if j in end_list:
                continue
            ed = pt[j + 1, :2]
            image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 1)
    return image

def plot_light(image, kpt, Y):
    image = image.copy()
    # kpt = np.round(kpt).astype(np.int32)
    kpt = np.array(kpt)
    # print(kpt.shape, type(kpt))
    # kpt = np.squeeze(kpt, axis=0)
    # kpt = np.transpose(kpt, (2, 1))

    # print("KPT shape", kpt.shape)
    for k in range(kpt.shape[0]):
        pt = kpt[k]
        # print(pt.shape)
        for i in range(len(matched)):
            x = int(pt[0, matched[i][1]])
            y = int(pt[1, matched[i][1]])
            # print(Y.shape, x, y)
            if x >= Y.shape[3] or y >= Y.shape[2]:
                continue
            light_color = Y[0, 0, y, x].item()
            light_color = int(light_color * 255)
            # print(light_color, type(light_color))
            # light_color = cv2.cvtColor(light_color * 255, cv2.COLOR_GRAY2BGR)
            image = cv2.circle(image, (x, y), 1, [light_color] * 3, 4)

    return image

def plot_mesh_simple(image, kpt):
    image = image.copy()

    # matched_dict = dict(matched)
    triangles = np.array(simple_tri)

    kpt = np.array(kpt)
    # print(kpt.shape, type(kpt))
    # kpt = np.squeeze(kpt, axis=0)
    # kpt = np.transpose(kpt, (2, 1))

    # print("KPT shape", kpt.shape, triangles.shape)

    for k in range(kpt.shape[0]):
        for i in range(triangles.shape[0]):
            idx0, idx1, idx2 = triangles[i][1] - 1, triangles[i][0] - 1, triangles[i][2] - 1
            # print("IDX 0 = ", idx0)
            idx0 = matched[idx0][1]
            # print("MATCHED IDX 0 = ", idx0)
            idx1 = matched[idx1][1]
            idx2 = matched[idx2][1]
            vv = kpt[k]
            p0 = (vv[0, idx0], vv[1, idx0])
            p1 = (vv[0, idx1], vv[1, idx1])
            p2 = (vv[0, idx2], vv[1, idx2])

            image = cv2.line(image, (p0[0], p0[1]), (p1[0], p1[1]), (255, 255, 255), 1)
            image = cv2.line(image, (p0[0], p0[1]), (p2[0], p2[1]), (255, 255, 255), 1)
            image = cv2.line(image, (p2[0], p2[1]), (p1[0], p1[1]), (255, 255, 255), 1)

    return image

def plot_mesh(image, kpt, triangles):
    image = image.copy()
    # kpt = np.round(kpt).astype(np.int32)
    kpt = np.array(kpt)
    print(kpt.shape, type(kpt))
    # kpt = np.squeeze(kpt, axis=0)
    # kpt = np.transpose(kpt, (2, 1))

    print("KPT shape", kpt.shape, triangles.shape)

    for k in range(kpt.shape[0]):
        for i in range(triangles.shape[1]):
            # s = 'f {} {} {}\n'.format(triangles[0, i], triangles[1, i], triangles[2, i])
            idx0, idx1, idx2 = triangles[1, i] - 1, triangles[0, i] - 1, triangles[2, i] - 1
            vv = kpt[k]
            p0 = (vv[0, idx0], vv[1, idx0])
            p1 = (vv[0, idx1], vv[1, idx1])
            p2 = (vv[0, idx2], vv[1, idx2])

            image = cv2.line(image, (p0[0], p0[1]), (p1[0], p1[1]), (255, 255, 0), 1)
            image = cv2.line(image, (p0[0], p0[1]), (p2[0], p2[1]), (255, 255, 0), 1)
            image = cv2.line(image, (p2[0], p2[1]), (p1[0], p1[1]), (255, 255, 0), 1)

    return image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.tif'])

def main(args):
    # 1. load pre-tained model
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    # 2. load dlib model for face detection and landmark used for face cropping
    if args.dlib_landmark:
        dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
        face_regressor = dlib.shape_predictor(dlib_landmark_model)
    if args.dlib_bbox:
        face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    filenames = [join(args.files, x) for x in listdir(args.files) if is_image_file(x)]
    filenames.sort()

    result_fn = join(args.files, 'results')
    if not os.path.exists(result_fn):
        os.mkdir(result_fn)

    blur = GaussianBlur_CUDA(sigma=35)
    rgb2y = RGB2Y('cuda')

    for img_fp in filenames:
        print("process ", img_fp)
        img_ori = cv2.imread(img_fp)
        img_data = torch.Tensor(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)).permute(2, 0, 1) / 255.0
        # print(img_data.shape)

        Y = blur(rgb2y.do(img_data.to('cuda').unsqueeze(0)))

        if args.dlib_bbox:
            rects = face_detector(img_ori, 1)
        else:
            rects = []

        if len(rects) == 0:
            rects = dlib.rectangles()
            rect_fp = img_fp + '.bbox'
            lines = open(rect_fp).read().strip().split('\n')[1:]
            for l in lines:
                l, r, t, b = [int(_) for _ in l.split(' ')[1:]]
                rect = dlib.rectangle(l, r, t, b)
                rects.append(rect)

        pts_res = []
        Ps = []  # Camera matrix collection
        poses = []  # pose collection, [todo: validate it]
        vertices_lst = []  # store multiple face vertices
        ind = 0
        suffix = get_suffix(img_fp)
        for rect in rects:
            # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
            if args.dlib_landmark:
                # - use landmark for cropping
                pts = face_regressor(img_ori, rect).parts()
                pts = np.array([[pt.x, pt.y] for pt in pts]).T
                roi_box = parse_roi_box_from_landmark(pts)
            else:
                # - use detected face bbox
                bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                roi_box = parse_roi_box_from_bbox(bbox)

            img = crop_img(img_ori, roi_box)

            # forward: one step
            img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input = input.cuda()
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            # 68 pts
            pts68 = predict_68pts(param, roi_box)

            # two-step for more accurate bbox to crop face
            if args.bbox_init == 'two':
                roi_box = parse_roi_box_from_landmark(pts68)
                img_step2 = crop_img(img_ori, roi_box)
                img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'gpu':
                        input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                pts68 = predict_68pts(param, roi_box)

            # print(pts68, len(pts68))
            pts_res.append(pts68)
            P, pose = parse_pose(param)
            Ps.append(P)
            poses.append(pose)

            # dense face 3d vertices
            if args.dump_ply or args.dump_vertex or args.dump_depth or args.dump_pncc or args.dump_obj or args.dump_res:
                vertices = predict_dense(param, roi_box)
                # print('vertex size is = ', len(vertices), vertices.shape, "param len ", len(param))
                vertices_lst.append(vertices)
                # print(param)
            if args.dump_ply:
                dump_to_ply(vertices, tri, '{}_{}.ply'.format(img_fp.replace(suffix, ''), ind))
            if args.dump_vertex:
                dump_vertex(vertices, '{}_{}.mat'.format(img_fp.replace(suffix, ''), ind))
            if args.dump_pts:
                wfp = '{}_{}.txt'.format(img_fp.replace(suffix, ''), ind)
                np.savetxt(wfp, pts68, fmt='%.3f')
                print('Save 68 3d landmarks to {}'.format(wfp))
            if args.dump_roi_box:
                wfp = '{}_{}.roibox'.format(img_fp.replace(suffix, ''), ind)
                np.savetxt(wfp, roi_box, fmt='%.3f')
                print('Save roi box to {}'.format(wfp))
            if args.dump_paf:
                wfp_paf = '{}_{}_paf.jpg'.format(img_fp.replace(suffix, ''), ind)
                wfp_crop = '{}_{}_crop.jpg'.format(img_fp.replace(suffix, ''), ind)
                paf_feature = gen_img_paf(img_crop=img, param=param, kernel_size=args.paf_size)

                cv2.imwrite(wfp_paf, paf_feature)
                cv2.imwrite(wfp_crop, img)
                print('Dump to {} and {}'.format(wfp_crop, wfp_paf))
            if args.dump_obj:
                wfp = '{}_{}.obj'.format(img_fp.replace(suffix, ''), ind)
                colors = get_colors(img_ori, vertices)
                write_obj_with_colors(wfp, vertices, tri, colors)
                print('Dump obj with sampled texture to {}'.format(wfp))
            ind += 1

        if args.dump_pose:
            # P, pose = parse_pose(param)  # Camera matrix (without scale), and pose (yaw, pitch, roll, to verify)
            img_pose = plot_pose_box(img_ori, Ps, pts_res)
            wfp = img_fp.replace(suffix, '_pose.jpg')
            cv2.imwrite(wfp, img_pose)
            print('Dump to {}'.format(wfp))
        if args.dump_depth:
            wfp = img_fp.replace(suffix, '_depth.png')
            # depths_img = get_depths_image(img_ori, vertices_lst, tri-1)  # python version
            depths_img = cget_depths_image(img_ori, vertices_lst, tri - 1)  # cython version
            cv2.imwrite(wfp, depths_img)
            print('Dump to {}'.format(wfp))
        if args.dump_pncc:
            wfp = img_fp.replace(suffix, '_pncc.png')
            pncc_feature = cpncc(img_ori, vertices_lst, tri - 1)  # cython version
            cv2.imwrite(wfp, pncc_feature[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR
            print('Dump to {}'.format(wfp))
        if args.dump_res:
            # print(pts_res, len(pts_res), len(pts_res[0][0]))
            # draw_landmarks(img_ori, pts_res, wfp=img_fp.replace(suffix, '_3DDFA.jpg'), show_flg=args.show_flg, color='yellow')

            # print("dump res")
            h, w = img_ori.shape[:2]
            blank = np.zeros((h, w, 3))
            result = plot_kpt(blank, pts_res)
            # print(result.shape)
            bn = basename(img_fp)
            ofn = join(result_fn, bn).replace(suffix, '_landmark.jpg')
            cv2.imwrite(ofn, result)

            # tri2 = tri / 29
            # print(tri2, tri)
            result = plot_mesh_simple(blank, vertices_lst)

            ofn = join(result_fn, bn).replace(suffix, '_mesh.jpg')
            cv2.imwrite(ofn, result)

            result = plot_light(blank, vertices_lst, Y)

            ofn = join(result_fn, bn).replace(suffix, '_light.jpg')
            cv2.imwrite(ofn, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-f', '--files', # nargs='+',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', default='false', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_res', default='true', type=str2bool, help='whether write out the visualization image')
    parser.add_argument('--dump_vertex', default='false', type=str2bool,
                        help='whether write out the dense face vertices to mat')
    parser.add_argument('--dump_ply', default='false', type=str2bool)
    parser.add_argument('--dump_pts', default='false', type=str2bool)
    parser.add_argument('--dump_roi_box', default='false', type=str2bool)
    parser.add_argument('--dump_pose', default='false', type=str2bool)
    parser.add_argument('--dump_depth', default='false', type=str2bool)
    parser.add_argument('--dump_pncc', default='false', type=str2bool)
    parser.add_argument('--dump_paf', default='false', type=str2bool)
    parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
    parser.add_argument('--dump_obj', default='false', type=str2bool)
    parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
    parser.add_argument('--dlib_landmark', default='true', type=str2bool,
                        help='whether use dlib landmark to crop image')

    args = parser.parse_args()
    main(args)
