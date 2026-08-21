# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import hashlib
import os
import os.path as osp
from os.path import join, exists, isdir, basename, abspath
from glob import glob
from tqdm import tqdm
import random
import cv2
import numpy as np
import json
import warnings
import math

import torch
import torch.nn.functional as F

import sys

backend_root = abspath(join(osp.dirname(__file__), '..'))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)
from utils.general import gaussian2D

sam_checkpoint = join(backend_root, "weights", "sam_vit_h_4b8939.pth")
model_type = "vit_h"
sam_load_error = None

try:
    from segment_anything import SamPredictor, sam_model_registry

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device) #.half()  Warning: Precision Drops
    dtype = next(sam.named_parameters())[1].dtype
    predictor = SamPredictor(sam)
except Exception as exc:
    sam_load_error = f'{type(exc).__name__}: {exc}'
    warnings.warn(
        'Segment Anything could not be loaded; hybrid pseudo masks are unavailable. '
        f'Reason: {sam_load_error}'
    )
    predictor = None


############ utils ############

def _readlines(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    
    return lines


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_break(act):
    flag = False
    prev_pos, prev_neg = False, False
    for x in act:
        if x:
            if not prev_pos:
                prev_pos = True
            elif prev_neg:
                flag = True
                break
        elif prev_pos:
            prev_neg = True
    
    return flag


def check_center(crop):
    h, w = crop.shape
    indices = torch.nonzero(crop)
    yc, xc = indices.float().mean(dim=0)
    
    s = 0.15
    return ((yc - h/2.).abs() > h * s) | ((xc - w/2.).abs() > w * s)


def segment_image(image, labels, width, height):
    if len(labels) == 0:
        return torch.zeros(image.shape[:2], dtype=torch.float16).to(device), np.full((0,), False)

    if max(width, height) > 1024:
        mask = torch.zeros((height, width), dtype=torch.float16).to(device)
        invalid = np.full((len(labels),), False)

        # overlap = 20  # pixel
        nx, ny = math.ceil(width / 1024), math.ceil(height / 1024)
        width_, height_ = width // nx, height // ny
        xc, yc, w, h = labels[:, -4:].T
        x1, y1, x2, y2 = xc - w / 2., yc - h / 2., xc + w / 2., yc + h / 2.
        for j in range(ny):
            for i in range(nx):
                grid = np.array([i / nx, j / ny, (i+1) / nx, (j+1) / ny], dtype=labels.dtype)
                indices = (grid[0] < x2) & (x1 < grid[2]) & (grid[1] < y2) & (y1 < grid[3])
                if indices.sum() == 0:
                    continue
                
                x1_, y1_, x2_, y2_ = (x1[indices] - grid[0]).clip(0, 1/nx), \
                                        (y1[indices] - grid[1]).clip(0, 1/ny), \
                                        (x2[indices] - grid[0]).clip(0, 1/nx), \
                                        (y2[indices] - grid[1]).clip(0, 1/ny)
                xc_, yc_, w_, h_ = (x1_ + x2_) / 2., (y1_ + y2_) / 2., (x2_ - x1_), (y2_ - y1_)
                labels_ = np.stack((labels[indices, 0], xc_, yc_, w_, h_), axis=1)

                x1_, y1_, x2_, y2_ = width_*i, height_*j, width_*(i+1), height_*(j+1)
                mask_k, invalid_k = segment_image(image[y1_:y2_, x1_:x2_], labels_, width_, height_)
                mask[y1_:y2_, x1_:x2_] = mask_k
                invalid[indices] |= invalid_k

        return mask, invalid
    
    c, xc, yc, w, h = labels.T
    x1, y1, x2, y2 = (xc - w / 2.) * width, (yc - h / 2.) * height, \
                    (xc + w / 2.) * width, (yc + h / 2.) * height
    input_boxes = np.stack((x1, y1, x2, y2), axis=1)
    input_boxes = torch.from_numpy(input_boxes).long().to(device)

    predictor.set_image(image)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2]).to(dtype)
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
        return_logits=True
    )
    # (batch_size) x (num_predicted_masks_per_input=1) x H x W
    mask = masks.sigmoid().squeeze(1).max(dim=0)[0].half()
    
    invalid = np.full((len(masks),), False)
    for i, (x1, y1, x2, y2) in enumerate(input_boxes.cpu()):
        crop = mask[y1:y2, x1:x2] > 0.5
        invalid[i] = check_center(crop) | check_break(crop.sum(dim=1)) | check_break(crop.sum(dim=0))
    
    return mask, invalid


def gen_mask(label_path, image, cls_ratio=False, thresh=0.5, sam_only=False, sam_mode='auto'):
    if sam_mode not in ('auto', 'gaussian', 'released-hybrid', 'paper-hybrid'):
        raise ValueError(f'unsupported sam_mode: {sam_mode}')
    use_sam = predictor is not None if sam_mode == 'auto' else sam_mode != 'gaussian'
    if use_sam and predictor is None:
        raise RuntimeError(
            f'{sam_mode} pseudo masks require Segment Anything and '
            f'{sam_checkpoint}; refusing to silently fall back to Gaussian masks'
        )
    if cls_ratio:
        cls_ratio = [1.83, 5.35, 13.82, 1.00, 5.80, 11.25, 30.11, 44.63, 24.45, 4.89]  # train set
    stride = 1
    # area_min, area_max = 4 * 4 * stride * stride, 6 * 6 * stride * stride
    area_min, area_max = 4 * 4 * 100, 6 * 6 * 50  # for 1920*1080
    min_size = 1e6

    save_path = label_path.replace('/labels/', '/masks/').replace('.txt', '.npy')
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    
    height, width, _ = image.shape
    nx, ny = width // stride, height // stride

    labels = np.loadtxt(label_path, delimiter=' ').reshape(-1, 5)

    mask = np.zeros((ny, nx), dtype=np.float16)
    weight = np.ones_like(mask)
    
    if use_sam:
        sam_res, invalid = segment_image(image, labels, width, height)
        if stride != 1:
            sam_res = F.interpolate(sam_res[None, None, ...].float(), size=(ny, nx), mode='bilinear', align_corners=False)[0, 0]
            # sam_res = F.interpolate(sam_res[None, None, ...].float(), size=(ny, nx), mode='nearest')[0, 0]
        # HESOD local patch: sam_res is a CUDA tensor (segment_image() runs SAM
        # inference on GPU) -- torch.Tensor.numpy() refuses to convert a CUDA
        # tensor directly ("can't convert cuda:0 device type tensor to numpy"),
        # needs an explicit .cpu() first. Upstream never hit this because it
        # presumably ran in an environment/PyTorch version lenient enough not
        # to raise, or was never actually exercised end-to-end.
        sam_res = (sam_res > 0.5).half().cpu().numpy()

    c, xc, yc, w, h = labels.T
    x1, y1, x2, y2 = ((xc - w / 2.) * nx).astype(np.int32).clip(0), \
                        ((yc - h / 2.) * ny).astype(np.int32).clip(0), \
                        np.ceil((xc + w / 2.) * nx).astype(np.int32).clip(0, nx), \
                        np.ceil((yc + h / 2.) * ny).astype(np.int32).clip(0, ny)
    input_boxes = np.stack((x1, y1, x2, y2), axis=1)
    
    for i, (x1, y1, x2, y2) in enumerate(input_boxes):
        w, h = x2 - x1, y2 - y1
        gaussian = gaussian2D((h, w), sigma=None, thresh=thresh).astype(mask.dtype)
        
        if use_sam:
            sam_mask = sam_res[y1:y2, x1:x2].copy()
            if sam_only:
                gaussian = sam_mask
            elif sam_mode == 'paper-hybrid':
                # ESOD Eq. (4): M = M_S * M_G when SAM is non-empty,
                # otherwise use the Gaussian mask unchanged.
                if sam_mask.sum() > 0:
                    gaussian *= sam_mask
            else:
                # Preserve the released implementation as a separately named
                # protocol. It adds a thresholded SAM mask back after the
                # product and therefore is not identical to paper Eq. (4).
                if invalid[i] == 0 and sam_mask.sum() / (w * h) > 0.25:
                    gaussian *= sam_mask
                np.maximum(gaussian, sam_mask * thresh, out=gaussian)
        
        masked_hm = mask[y1:y2, x1:x2]
        np.maximum(masked_hm, gaussian, out=masked_hm)

        area = w * h / (width * height) * (1920 * 1080)
        # min_size = min(min_size, area)
        r_size = max(area_min / area, 1.) ** 2
        # elif area > area_max:
        #     r_size = area / area_max
        if cls_ratio:
            r_cls = cls_ratio[int(c[i])] ** 0.7
        else:
            r_cls = 1.0
        r = max(r_size, r_cls)
        masked_wt = weight[y1:y2, x1:x2]
        curr_wt = np.zeros_like(masked_wt) + math.log(r) + 1.
        curr_wt *= (gaussian > 0).astype(mask.dtype)
        np.maximum(masked_wt, curr_wt, out=masked_wt)
    
    np.save(save_path, np.stack((mask, weight), axis=-1))


############ scripts ############


def prepare_visdrone():
    
    name_dict = {'0': 'ignored regions', '1': 'pedestrian', '2': 'people',
                 '3': 'bicycle', '4': 'car', '5': 'van', '6': 'truck',
                 '7': 'tricycle', '8': 'awning-tricycle', '9': 'bus',
                 '10': 'motor', '11': 'others'}
    split_dict = {'test-dev': 'test-dev.txt', 'val': 'val.txt', 'train': 'train.txt'}
    root = opt.dataset

    os.makedirs(join(root, 'split'), exist_ok=True)
    for sub_dir in glob(join(root, 'VisDrone2019-DET-*')):
        os.makedirs(join(sub_dir, 'labels'), exist_ok=True)
        images = sorted(glob(join(sub_dir, 'images', '*.jpg')))
        if 'test-challenge' in sub_dir:
            with open(join(root, 'split', 'test-challenge.txt'), 'w+') as f:
                f.writelines([line + '\n' for line in images])
            continue

        data_paths = []
        for image_path in tqdm(images):
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            label_path = image_path.replace('images', 'annotations').replace('.jpg', '.txt')
            # if "masked" in label_path:  # avoid repeated processing
            #     continue
            assert exists(label_path)
            label_lines = []
            masked = False
            # <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
            for line in _readlines(label_path):
                if line[-1] == ',':
                    line = line[:-1]
                x1, y1, w, h, score, cls, truncation, occlusion = list(map(int, line.split(',')))
                if cls in [0, 11]:
                    image[y1:y1 + h, x1:x1 + w, :] = 85
                    masked = True
                elif truncation < 2 and (occlusion < 2 or True):
                    xc, yc = x1 + w / 2., y1 + h / 2.
                    label_lines.append(('%d' + ' %.6f' * 4 + '\n') %
                                       (cls - 1, xc / width, yc / height, w / width, h / height))
                else:
                    pass
            if masked:
                image_path = image_path.replace('.jpg', '_masked.jpg')
                cv2.imwrite(image_path, image)
            # for consistency
            label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
            with open(label_path, 'w+') as f:
                f.writelines(label_lines)

            gen_mask(label_path, image, cls_ratio=True)

            data_paths.append(image_path + '\n')
        
        with open(join(root, 'split', split_dict[basename(sub_dir)[17:]]), 'w+') as f:
            f.writelines(data_paths)


def prepare_uavdt():
    root = opt.dataset
    data_dir = join(root, 'UAV-benchmark-M')
    attr_dir = join(root, 'M_attr')
    label_dir = join(root, 'UAV-benchmark-MOTD_v1.0', 'GT')
    split_dir = join(root, 'split')
    
    # mask images, it takes minutes
    for i, path in enumerate(glob(join(label_dir, '*ignore.txt'))):
        labels = np.loadtxt(path, usecols=(0, 2, 3, 4, 5, 8), dtype=int, delimiter=',')
        vid_name = basename(path).split('_')[0]
        for frameID, x1, y1, w, h, _ in tqdm(labels, desc='%02d/50' % (i + 1)):
            masked_path = join(data_dir, vid_name, 'img%06d_masked.jpg' % frameID)
            input_path = masked_path if exists(masked_path) else masked_path.replace('_masked.jpg', '.jpg')
            image = cv2.imread(input_path)
            image[y1:y1 + h, x1:x1 + w] = (127, 127, 127)
            cv2.imwrite(masked_path, image)

    data_split = {}
    for mode in ['train', 'test']:
        data_split[mode] = []
        for path in glob(join(attr_dir, mode, '*.txt')):
            data_split[mode].append(basename(path)[:5])
    k = 10
    sep = len(data_split['train']) // k
    random.shuffle(data_split['train'])
    data_split['train'], data_split['valid'] = data_split['train'][sep:], data_split['train'][:sep]

    os.makedirs(split_dir, exist_ok=True)
    for mode in ['train', 'valid', 'test']:
        with open(join(split_dir, '%s_video.txt' % mode), 'w+') as f:
            f.writelines(vid + '\n' for vid in data_split[mode])
        image_paths = []
        for video_name in tqdm(data_split[mode], desc=mode):
            ignore_path = join(label_dir, '%s_gt_ignore.txt' % video_name)
            label_path = join(label_dir, '%s_gt_whole.txt' % video_name)
        
            # the warning caused by empty file doesn't matter
            ignores = np.loadtxt(ignore_path, usecols=(0, 2, 3, 4, 5), dtype=int, delimiter=',')
            ignore_dict = {}
            for frameID, x1, y1, w, h in ignores:
                xyxy = np.array([[x1, y1, x1 + w, y1 + h]])
                if frameID in ignore_dict:
                    ignore_dict[frameID] = np.concatenate((ignore_dict[frameID], xyxy), axis=0)
                else:
                    ignore_dict[frameID] = xyxy
        
            labels = np.loadtxt(label_path, usecols=(0, 2, 3, 4, 5, 8), dtype=int, delimiter=',')
            label_dict = {}
            for frameID, x1, y1, w, h, cls in labels:
                xc, yc = x1 + w / 2., y1 + h / 2.
                if frameID in ignore_dict:
                    ignore_regions = ignore_dict[frameID]
                    if np.logical_and(
                            np.logical_and(ignore_regions[:, 0] < xc, xc < ignore_regions[:, 2]),
                            np.logical_and(ignore_regions[:, 1] < yc, yc < ignore_regions[:, 3])
                    ).sum() > 0:
                        continue
            
                box = [cls, xc, yc, w, h]
                if frameID in label_dict:
                    label_dict[frameID].append(box)
                else:
                    label_dict[frameID] = [box]
        
            for frameID, bboxes in label_dict.items():
                image_path = join(data_dir, video_name, 'img%06d_masked.jpg' % frameID)
                if not exists(image_path):
                    image_path = image_path.replace('_masked.jpg', '.jpg')
            
                image = cv2.imread(image_path)
                height, width, _ = image.shape
                label_path = image_path.replace('.jpg', '.txt')
                with open(label_path, 'w+') as f:
                    for cls, xc, yc, w, h in bboxes:
                        assert 1 <= cls <= 3  # cls - 1
                        # Paper-comparable UAVDT-DET keeps car/truck/bus as three
                        # classes. The released converter collapsed everything to
                        # class 0, producing a much easier, non-comparable task.
                        out_cls = int(cls) - 1
                        f.write('%d %.6f %.6f %.6f %.6f\n' % (out_cls, xc / width, yc / height, w / width, h / height))

                gen_mask(label_path, image)
                image_paths.append(image_path + '\n')
    
        with open(join(split_dir, '%s.txt' % mode), 'w+') as f:
            f.writelines(image_paths)
        if mode == 'train':
            with open(join(split_dir, '%s_ds.txt' % mode), 'w+') as f:
                f.writelines(image_paths[::10])


def prepare_tinyperson():
    # TinyPerson's benchmark protocol excludes dense images from both training
    # and evaluation and uses images with ignore/uncertain regions erased.
    # The paper's 794/816 figures describe the complete dataset; they are not
    # the benchmark train/evaluation split. Never substitute the with_dense
    # JSONs or raw image tree here.
    root = opt.dataset
    label_file_dict = {'train': join(root, 'mini_annotations', 'tiny_set_train_all_erase.json'),
                       'test': join(root, 'mini_annotations', 'tiny_set_test_all.json')}
    image_dir = join(root, 'erase_with_uncertain_dataset')
    split_dir = join(root, 'split')
    mask_mode = opt.tinyperson_mask_mode

    if mask_mode != 'gaussian' and predictor is None:
        raise RuntimeError(
            f'--tinyperson-mask-mode={mask_mode} requires Segment Anything and '
            f'{sam_checkpoint}. SAM load error: {sam_load_error}. '
            'Install/load it or explicitly select gaussian.'
        )

    os.makedirs(split_dir, exist_ok=True)
    split_stats = {}
    for mode in ['train', 'test']:
        with open(label_file_dict[mode], 'r') as f:
            anno = json.load(f)

        image_dict = {}
        for item in anno['images']:
            file_name, width, height = item['file_name'], item['width'], item['height']
            file_path = join(image_dir, mode, file_name)
            image_dict[item['id']] = {'shape': [width, height], 'bboxes': [], 'image_path': file_path}

        missing_images = [
            item['image_path'] for item in image_dict.values()
            if not exists(item['image_path'])
        ]
        if missing_images:
            samples = '\n'.join(f'  - {path}' for path in missing_images[:5])
            raise FileNotFoundError(
                f'TinyPerson erased-image layout does not match {label_file_dict[mode]}. '
                f'Missing {len(missing_images)}/{len(image_dict)} {mode} images. '
                'Expected paths such as '
                f'{join(image_dir, mode, "labeled_images", "<file>.jpg")}.\n'
                f'First missing paths:\n{samples}\n'
                'Do not substitute the un-erased train/test images: that changes '
                'the official TinyPerson protocol.'
            )
        
        for item in anno['annotations']:
            if item.get('ignore', False) or item.get('uncertain', False):
                continue
            _id, (x1, y1, w, h) = item['image_id'], item['bbox']
            (width, height) = image_dict[_id]['shape']
            xc, yc, w, h = (x1 + w / 2.) / width, (y1 + h / 2.) / height, w / width, h / height
            image_dict[_id]['bboxes'].append('0 %.6f %.6f %.6f %.6f\n' % (xc, yc, w, h))
    
        paths = []
        for item in image_dict.values():
            image_path = item['image_path']
            label_path = item['image_path'][:-4] + '.txt'
            with open(label_path, 'w+') as f:
                f.writelines(item['bboxes'])
            
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f'failed to read TinyPerson image: {image_path}')
            gen_mask(label_path, image, sam_mode=mask_mode)
            paths.append(image_path + '\n')

        split_stats[mode] = {
            'images': len(paths),
            'positive_boxes': sum(len(item['bboxes']) for item in image_dict.values()),
        }
        
        if mode == 'train':
            with open(join(split_dir, 'trainval.txt'), 'w+') as f:
                f.writelines(paths)
            k = 10
            random.shuffle(paths)
            sep = len(paths) // k
            with open(join(split_dir, 'train.txt'), 'w+') as f:
                f.writelines(paths[sep:])
            with open(join(split_dir, 'valid.txt'), 'w+') as f:
                f.writelines(paths[:sep])
        else:
            with open(join(split_dir, 'test.txt'), 'w+') as f:
                f.writelines(paths)

    manifest = {
        'schema_version': 1,
        'protocol': 'tinyperson-official-no-dense-erased',
        'dense_images': False,
        'erased_ignore_uncertain_pixels': True,
        'mask_mode': mask_mode,
        'annotations': {
            mode: {
                'path': osp.relpath(path, root).replace('\\', '/'),
                'sha256': _sha256(path),
            }
            for mode, path in label_file_dict.items()
        },
        'splits': split_stats,
    }
    with open(join(split_dir, 'tinyperson_protocol.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
                

def prepare_seadronesseev2():
    # SeaDronesSeeV2 "CompressedVersion" download: already COCO-format JSON
    # (annotations/instances_{train,val}.json), flat images/{split}/*.jpg,
    # no ignore/uncertain annotation fields (confirmed 2026-08-xx diagnostic --
    # annotation keys are only bbox/image_id/id/area/category_id). category_id
    # 0 ('ignored') is defined in the taxonomy but never actually used in any
    # annotation, so no filtering is needed there either -- every annotation
    # converts directly. category_id 1-5 -> yolo class 0-4
    # (swimmer/boat/jetski/life_saving_appliances/buoy, matching nc=5 in
    # seadronesseev2_yolov5m*.yaml). images/test/ has no instances_test.json
    # (ground truth is held by the SeaDronesSee benchmark server, per the
    # paper) -- only train/val are converted; data yaml's test: key points at
    # the same split/val.txt as val: so --task test still works unmodified.
    root = opt.dataset
    cat_id_to_yolo = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    split_dict = {'train': 'instances_train.json', 'val': 'instances_val.json'}
    split_dir = join(root, 'split')
    os.makedirs(split_dir, exist_ok=True)

    for mode, ann_file in split_dict.items():
        with open(join(root, 'annotations', ann_file), 'r') as f:
            anno = json.load(f)

        image_dir = join(root, 'images', mode)
        label_dir = join(root, 'labels', mode)
        os.makedirs(label_dir, exist_ok=True)

        image_dict = {}
        for item in anno['images']:
            file_name, width, height = item['file_name'], item['width'], item['height']
            image_dict[item['id']] = {
                'shape': [width, height],
                'bboxes': [],
                'image_path': join(image_dir, file_name),
                'label_path': join(label_dir, osp.splitext(file_name)[0] + '.txt'),
            }

        for item in anno['annotations']:
            _id, (x1, y1, w, h) = item['image_id'], item['bbox']
            cls = cat_id_to_yolo[item['category_id']]
            width, height = image_dict[_id]['shape']
            xc, yc = (x1 + w / 2.) / width, (y1 + h / 2.) / height
            image_dict[_id]['bboxes'].append(
                ('%d' + ' %.6f' * 4 + '\n') % (cls, xc, yc, w / width, h / height))

        paths = []
        for item in tqdm(image_dict.values(), desc=f'seadronesseev2/{mode}'):
            with open(item['label_path'], 'w+') as f:
                f.writelines(item['bboxes'])

            image = cv2.imread(item['image_path'])
            gen_mask(item['label_path'], image)
            paths.append(item['image_path'] + '\n')

        with open(join(split_dir, f'{mode}.txt'), 'w+') as f:
            f.writelines(paths)


def prepare_seaperson():
    # SeaPerson (aka TinyPersonV2) whole-image release: release/rgb_{train,
    # valid,test}.json under anns_extracted/ (release/corner/*.json are
    # pre-tiled sliding-window variants, excluded -- same reasoning as
    # TinyPerson's own excluded sw640_sh512 variant: ESOD's method self-routes
    # full images, it does not want pre-cropped tiles). valid.json is the
    # official held-out split (confirmed train_ids | valid_ids ==
    # trainvalid_ids exactly, zero train/test id overlap, 2026-08-xx
    # diagnostic) -- trainvalid.json itself is never read here.
    #
    # Same annotation schema as TinyPerson's with_dense format (image/
    # annotation keys verified identical, including ignore/uncertain/
    # in_dense_image); file_name already includes the 'rgb/...' relative
    # path, so images resolve as join(root, file_name) with imgs_rgb.zip
    # extracted directly under root.
    #
    # Unlike TinyPerson, this release ships no erase-preprocessed image
    # variant -- only raw imgs_rgb.zip. Mirrors prepare_visdrone()'s existing
    # ignored-region erasing convention (mid-gray fill, write only if any
    # erasing occurred) rather than doing label-level-only ignore/uncertain
    # filtering on unmodified pixels, so ambiguous regions don't leak into
    # other boxes' negative-space supervision the way TinyPerson's protocol
    # was specifically reworked to avoid.
    #
    # Labels/masks are written in-place next to each (possibly erased) image,
    # same convention as prepare_uavdt()/prepare_tinyperson() -- SeaPerson's
    # ~300 per-video-sequence subfolders are the same shape as UAVDT's
    # per-video layout. This in-place layout has no flat directory for
    # audit_buckets.py/vt_diagnose.py's --labels/--images convention (same
    # gap UAVDT hit) -- run scripts/esod_baseline/reorganize_seaperson.py
    # against this function's output before training/auditing.
    root = opt.dataset
    cat_id_to_yolo = {1: 0}
    label_file_dict = {
        'train': join(root, 'anns_extracted', 'release', 'rgb_train.json'),
        'valid': join(root, 'anns_extracted', 'release', 'rgb_valid.json'),
        'test': join(root, 'anns_extracted', 'release', 'rgb_test.json'),
    }
    split_dir = join(root, 'split')
    os.makedirs(split_dir, exist_ok=True)

    for mode, label_file in label_file_dict.items():
        with open(label_file, 'r') as f:
            anno = json.load(f)

        image_dict = {}
        for item in anno['images']:
            file_name, width, height = item['file_name'], item['width'], item['height']
            image_dict[item['id']] = {
                'shape': [width, height], 'bboxes': [], 'erase_boxes': [],
                'image_path': join(root, file_name),
            }

        missing_images = [
            item['image_path'] for item in image_dict.values()
            if not exists(item['image_path'])
        ]
        if missing_images:
            samples = '\n'.join(f'  - {path}' for path in missing_images[:5])
            raise FileNotFoundError(
                f'SeaPerson image layout does not match {label_file}. '
                f'Missing {len(missing_images)}/{len(image_dict)} {mode} images. '
                f'Expected imgs_rgb.zip extracted directly under {root} '
                "(so file_name's own 'rgb/...' prefix resolves without an "
                f'extra subdirectory).\nFirst missing paths:\n{samples}\n'
            )

        n_degenerate = 0
        for item in anno['annotations']:
            _id, (x1, y1, w, h) = item['image_id'], item['bbox']
            width, height = image_dict[_id]['shape']
            if item.get('ignore', False) or item.get('uncertain', False):
                image_dict[_id]['erase_boxes'].append((x1, y1, w, h))
                continue
            # A raw box with w,h >= 1px can still collapse to zero visible
            # area once gen_mask() clamps it to the actual image frame (a
            # handful of raw annotations extend past the image edge --
            # 262k boxes from varied scraped sources). Clamp here, against
            # the real frame, and drop what's left empty, rather than
            # letting a clamped zero-size box reach gaussian2D().
            x1c, y1c = max(x1, 0), max(y1, 0)
            x2c, y2c = min(x1 + w, width), min(y1 + h, height)
            if x2c - x1c < 1 or y2c - y1c < 1:
                n_degenerate += 1
                continue
            cls = cat_id_to_yolo[item['category_id']]
            xc, yc = (x1c + x2c) / 2. / width, (y1c + y2c) / 2. / height
            w_n, h_n = (x2c - x1c) / width, (y2c - y1c) / height
            image_dict[_id]['bboxes'].append(
                ('%d' + ' %.6f' * 4 + '\n') % (cls, xc, yc, w_n, h_n))
        if n_degenerate:
            print(f'seaperson/{mode}: skipped {n_degenerate} zero-area/out-of-frame annotation(s)')

        paths = []
        for item in tqdm(image_dict.values(), desc=f'seaperson/{mode}'):
            image_path = item['image_path']
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f'failed to read SeaPerson image: {image_path}')

            if item['erase_boxes']:
                for x1, y1, w, h in item['erase_boxes']:
                    x1, y1, w, h = int(x1), int(y1), int(w), int(h)
                    image[y1:y1 + h, x1:x1 + w, :] = 127
                root_name, ext = osp.splitext(image_path)
                image_path = root_name + '_erased' + ext
                cv2.imwrite(image_path, image)

            label_path = osp.splitext(image_path)[0] + '.txt'
            with open(label_path, 'w+') as f:
                f.writelines(item['bboxes'])

            gen_mask(label_path, image)
            paths.append(image_path + '\n')

        with open(join(split_dir, f'{mode}.txt'), 'w+') as f:
            f.writelines(paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='VisDrone', help='dataset, e.g., VisDrone, UAVDT, TinyPerson, and SeaDronesSeeV2')
    parser.add_argument(
        '--tinyperson-mask-mode',
        choices=('paper-hybrid', 'released-hybrid', 'gaussian'),
        default='paper-hybrid',
        help='TinyPerson pseudo-mask protocol; hybrid modes fail if SAM is unavailable',
    )
    opt = parser.parse_args()

    assert exists(opt.dataset)
    dataset = opt.dataset.lower()
    if 'visdrone' in dataset:
        prepare_visdrone()
    elif 'uavdt' in dataset:
        prepare_uavdt()
    elif 'tinyperson' in dataset:
        prepare_tinyperson()
    elif 'seadrones' in dataset or 'compressedversion' in dataset:
        prepare_seadronesseev2()
    elif 'seaperson' in dataset:
        prepare_seaperson()
    else:
        print('%s is coming soon.' % opt.dataset)
