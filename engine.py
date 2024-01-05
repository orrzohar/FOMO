# ------------------------------------------------------------------------
# Open World Object Detection in the Era of Foundation Models
# Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import os
import torch
import util.misc as utils
from datasets.open_world_eval import OWEvaluator
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np


@torch.no_grad()
def evaluate(model, postprocessors, data_loader, base_ds, device, output_dir, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    coco_evaluator = OWEvaluator(base_ds, args=args)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples.tensors)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)

        image_ids = [''.join([chr(int(t)) for t in target['image_id']]) for target in targets]
        if len(set(image_ids)) != len(image_ids):
            import ipdb;
            ipdb.set_trace()
        res = {''.join([chr(int(t)) for t in target['image_id']]): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        res = coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['metrics'] = res
    if coco_evaluator is not None:
        stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    return stats, coco_evaluator


@torch.no_grad()
def viz(model, postprocessors, data_loader, device, output_dir, base_ds, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Viz:'

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples.tensors)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessors(outputs, orig_target_sizes, args.viz)
        plot_batch(samples.tensors, results, targets, args.output_dir,
                   [''.join([chr(int(i)) for i in target['image_id']]) + '.jpg' for target in targets],
                   base_ds.KNOWN_CLASS_NAMES + ['unknown'], orig_target_sizes)

    return


@torch.no_grad()
def plot_batch(samples, results, targets, output_dir, image_names, cls_names, orig_target_sizes):
    for i, r in enumerate(results):
        img = samples[i].swapaxes(0, 1).swapaxes(1, 2).detach().cpu()
        plot_bboxes_on_image({k: v.detach().cpu() for k, v in r.items()}, img.numpy(), output_dir, image_names[i],
                             cls_names, num_known=sum(targets[i]['labels'] < len(cls_names) - 1),
                             num_unknown=sum(targets[i]['labels'] == len(cls_names) - 1), img_size=orig_target_sizes[i])

    return


def plot_bboxes_on_image(detections, img, output_dir, image_name, cls_names, num_known=10, num_unknown=5,
                         img_size=None):
    os.makedirs(output_dir, exist_ok=True)
    img = img * np.array([0.26862954, 0.26130258, 0.27577711]) + np.array([0.48145466, 0.4578275, 0.40821073])
    # Extract detections from dictionary
    # import ipdb; ipdb.set_trace()
    if True:
        unk_ind = detections['labels'] == len(cls_names) - 1
        unk_s = detections['scores'][unk_ind]
        unk_l = detections['labels'][unk_ind]
        unk_b = detections['boxes'][unk_ind]
        unk_s, indices = unk_s.topk(min(num_unknown + 1, len(unk_s)))
        unk_l = unk_l[indices]
        unk_b = unk_b[indices]

        k_s = detections['scores'][~unk_ind]
        k_l = detections['labels'][~unk_ind]
        k_b = detections['boxes'][~unk_ind]
        k_s, indices = k_s.topk(min(num_known + 3, len(k_s)))
        k_l = k_l[indices]
        k_b = k_b[indices]
        scores = torch.cat([k_s, unk_s])
        labels = torch.cat([k_l, unk_l])
        boxes = torch.cat([k_b, unk_b])
    else:
        scores = detections['scores']
        labels = detections['labels']
        boxes = detections['boxes']

    fig, ax = plt.subplots(1)
    plt.axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    img_size = (img_size * 840 / img_size.max()).detach().cpu().numpy().astype('int32')
    ax.imshow(img[:img_size[0], :img_size[1], :])

    # Plot bounding boxes on image
    for i in range(len(labels)):
        score = scores[i]
        label = cls_names[int(labels[i])]
        if (label == 'unknown' and score > -0.025) or \
                (label != 'unknown' and score > 0.25) or label == 'fish':

            box = boxes[i]

            xmin, ymin, xmax, ymax = [int(b) for b in box.numpy().astype(np.int32)]
            if label == 'unknown':
                rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='b', facecolor='none')
            else:
                rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='g', facecolor='none')

            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f'{label}: {score:.2f}', fontsize=10, color='g')
    plt.savefig(os.path.join(output_dir, image_name), dpi=300, bbox_inches='tight', pad_inches=0)
    return
