# ------------------------------------------------------------------------
# Open World Object Detection in the Era of Foundation Models
# Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang
# ------------------------------------------------------------------------
# Modified from PROB: Probabilistic Objectness for Open World Object Detection
# Orr Zohar, Jackson Wang, Serena Yeung
# ------------------------------------------------------------------------

import argparse
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
import pandas as pd
from engine import viz, evaluate
from models import build_model
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser('RWD - FOMO Detector', add_help=False)
    parser.add_argument('--batch_size', default=10, type=int)
    # dataset parameters
    parser.add_argument('--output_dir', default='tmp/rwod',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    ################ dataset configs ################
    parser.add_argument('--test_set', default='test.txt', help='testing txt files')
    parser.add_argument('--train_set', default='train.txt', help='training txt files')
    parser.add_argument('--dataset', default='?', help='defines which dataset is used.')
    parser.add_argument('--data_root', default='./data', type=str)
    parser.add_argument('--data_task', default='RWD', type=str)
    parser.add_argument('--unknown_classnames_file', default='', type=str)
    parser.add_argument('--classnames_file', default='known_classnames.txt', type=str)
    parser.add_argument('--prev_classnames_file', default='known_classnames.txt', type=str)
    parser.add_argument('--templates_file', default='best_templates.txt', type=str)
    parser.add_argument('--attributes_file', default='attributes.json', type=str)
    parser.add_argument('--pred_per_im', default=100, type=int)
    parser.add_argument('--PREV_INTRODUCED_CLS', default=0, type=int)
    parser.add_argument('--CUR_INTRODUCED_CLS', default=30, type=int)
    parser.add_argument('--image_conditioned_file', default='few_shot_data.json', type=str)

    ################ model configs ################
    parser.add_argument('--use_attributes', action='store_true')
    parser.add_argument('--att_selection', action='store_true')
    parser.add_argument('--att_refinement',  action='store_true')
    parser.add_argument('--att_adapt',  action='store_true')
    parser.add_argument('--post_process_method', default='regular',
                        help='seperated: Used for the fs baseline attributes: Used for attribute experiments')

    parser.add_argument('--image_conditioned', action='store_true')
    parser.add_argument('--num_few_shot', default=100, type=int)
    parser.add_argument('--num_att_per_class', default=25, type=int)
    parser.add_argument('--unk_methods', default='sigmoid-max-mcm', type=str)
    parser.add_argument('--unk_method', default='sigmoid-max-mcm', type=str)
    parser.add_argument('--model_name', default='google/owlvit-base-patch16', type=str)
    parser.add_argument('--unk_proposal', action='store_true')
    parser.add_argument('--image_resize', default=768, type=int,
                        help='image resize 768 for owlvit-base models, 840 for owlvit-large models')
    parser.add_argument('--prev_output_file', default='', type=str)
    parser.add_argument('--output_file', default='', type=str)

    parser.add_argument('--TCP', default='295499', type=str)
    return parser


def main(args):
    print(args)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ###### get datasets ######
    if len(args.train_set) > 0:
        dataset_train = build_dataset(args, args.train_set)
        data_loader_train = get_dataloader(args, dataset_train, train=False)

    if len(args.test_set) > 0:
        dataset_val = build_dataset(args, args.test_set)
        data_loader_val = get_dataloader(args, dataset_val, train=False)

    neg_sup_ep = [1, 10, 100]
    neg_sup_lr = [1e-5, 5e-5, 1e-4]
    best_kmap  = -1 
    bad        = 0

    if (len(neg_sup_ep) > 1 or len(neg_sup_lr) > 1) and args.image_conditioned and args.att_refinement:
        for eps in tqdm(neg_sup_ep, desc='Epochs', leave=False):
            for lr in tqdm(neg_sup_lr, desc='lr', leave=False):
                if bad > 2:
                    continue
                args.neg_sup_ep = eps
                args.neg_sup_lr = lr

                model, postprocessors = build_model(args)
                model.to(device)

                if args.distributed:
                    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

                test_stats, coco_evaluator = evaluate(model, postprocessors,
                                                      data_loader_train, dataset_train,
                                                      device, args.output_dir, args)

                if test_stats['metrics']['K_AP50'] > best_kmap:
                    best_kmap = test_stats['metrics']['K_AP50']
                    bad = 0
                    best_eps = eps
                    best_lr = lr
                else:
                    bad += 1

        args.neg_sup_ep = best_eps
        args.neg_sup_lr = best_lr

    else:
        args.neg_sup_ep = neg_sup_ep[0]
        args.neg_sup_lr = neg_sup_lr[0]

    model, postprocessors = build_model(args)
    model.to(device)

    if args.viz:
        viz(model, postprocessors, data_loader_val, device, args.output_dir, dataset_val, args)
        return

    unk_methods = args.unk_methods.split(",")
    for unk_method in unk_methods:
        print(f"\n running method {unk_method}\n")
        model.unk_head.method = unk_method

        test_stats, coco_evaluator = evaluate(model, postprocessors, data_loader_val, dataset_val, device,
                                              args.output_dir, args)
        output = test_stats['metrics']
        output.update({'model': args.model_name,
                       'dataset': args.dataset,
                       'unk_proposal': args.unk_proposal,
                       'unk_method': args.unk_method,
                       'classnames_file': args.classnames_file,
                       'unknown_classnames_file': args.unknown_classnames_file,
                       'pred_per_im': args.pred_per_im,
                       'num_few_shot': args.num_few_shot,
                       'templates_file': args.templates_file})

        output = pd.DataFrame(output, index=[0])
        if args.prev_output_file:
            try:
                tmp = pd.read_csv(f'{args.output_dir}/{args.prev_output_file}', index_col=0)
                output = pd.concat([tmp, output], ignore_index=True)
            except:
                print('previous file does not exist')

        if args.output_file:
            output_dir = Path(args.output_dir)
            if not output_dir.exists():
                os.makedirs(output_dir, exist_ok=True)
            output_path = output_dir / args.output_file
            output.to_csv(output_path)


def get_dataloader(args, dataset, train=True):
    if args.distributed:
        sampler = samplers.DistributedSampler(dataset, shuffle=train)
    else:
        if train:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    if train:
        batch_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)
        data_loader = DataLoader(dataset, batch_sampler=batch_sampler,
                                 collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    else:
        data_loader = DataLoader(dataset, args.batch_size, sampler=sampler,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    return data_loader


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RWOD and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    print("*********************************************Finshed Run*********************************************")