"""
This code is built upon DiffAct: https://github.com/Finspire13/DiffAct
"""
import os
import torch
import argparse
import numpy as np

from src.dataset import get_data_dict, VideoFeatureDataset
from src.utils import load_config_file, read_mapping_dict
from src.trainer import Trainer

import wandb
import random
from torch.backends import cudnn

# Seed fix 
#seed = 13452
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)
#cudnn.benchmark, cudnn.deterministic = False, True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--test', action='store_true', help='only test mode')
    parser.add_argument('--result_dir', type=str, default='actfusion')
    parser.add_argument('--ckpt', action='store_true', help='inference with checkpoint models')

    args = parser.parse_args()

    all_params = load_config_file(args.config)
    locals().update(all_params)
    
    # Add pos, n_mask, and patch_size from config to args
    args.pos = all_params.get('pos', 'none')
    args.n_mask = all_params.get('n_mask', 10)
    args.patch_size = all_params.get('patch_size', 10)

    naming = args.result_dir
    device = torch.device('cuda')

    if all_params['dataset_name'] == '50salads':
        wandb.init(project='50s_diffusion_integrate')
    elif all_params['dataset_name'] == 'gtea':
        wandb.init(project='gtea_diffusion_integrate')
    else:
        wandb.init(project='bf_diffusion_integrate')

    wandb.run.name = args.result_dir
    wandb.config.update(vars(args), allow_val_change=True)
    wandb.config.update(all_params, allow_val_change=True)

    feature_dir = os.path.join(root_data_dir, dataset_name, 'features')
    label_dir = os.path.join(root_data_dir, dataset_name, 'groundTruth')
    mapping_file = os.path.join(root_data_dir, dataset_name, 'mapping.txt')

    actions_dict = read_mapping_dict(mapping_file)

    event_list = np.loadtxt(mapping_file, dtype=str)
    event_list = [i[1] for i in event_list]
    num_classes = len(event_list)
    split = args.split
    print("split: ",split)

    train_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, 'splits', f'train.split{split}.bundle'), dtype=str)
    test_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, 'splits', f'test.split{split}.bundle'), dtype=str)

    train_video_list = [i.split('.')[0] for i in train_video_list]
    test_video_list = [i.split('.')[0] for i in test_video_list]

    if not args.test:
        train_data_dict = get_data_dict(
            feature_dir=feature_dir,
            label_dir=label_dir,
            video_list=train_video_list,
            event_list=event_list,
            sample_rate=sample_rate,
            temporal_aug=temporal_aug,
            boundary_smooth=boundary_smooth
        )
        train_train_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='train')
        train_test_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='test')

    test_data_dict = get_data_dict(
        feature_dir=feature_dir,
        label_dir=label_dir,
        video_list=test_video_list,
        event_list=event_list,
        sample_rate=sample_rate,
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth
    )

    test_test_dataset = VideoFeatureDataset(test_data_dict, num_classes, mode='test')

    trainer = Trainer(dict(encoder_params), dict(decoder_params), dict(diffusion_params),
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess,
        device=device, args=args
    )

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if args.test:
        device = torch.device('cuda')
        mode = 'decoder-agg'

        if args.ckpt:
            model_path = os.path.join('ckpt', dataset_name, 'split'+str(args.split)+'.model')
        else:
            model_path = os.path.join('result', args.result_dir, dataset_name, 'split'+str(args.split), 'best_combined_model.pth')
        print("model loaded:", model_path)
        result_path = os.path.join('result', args.result_dir, dataset_name, 'split'+str(args.split))

        # For test mode, always run both TAS and LTA inference
        print("TAS inference")
        test_result_dict = trainer.test(
            test_test_dataset, mode, device, label_dir,
            result_dir=result_path, model_path=model_path, args=args, all_params=all_params, obs_p=1.0)
        
        print("LTA inference")
        obs_ps = [0.2, 0.3]
        for obs_p in obs_ps:
            print("LTA inference: obs_p", obs_p)
            test_result_dict = trainer.test(
                test_test_dataset, mode, device, label_dir, args=args,
                result_dir=result_path, model_path=model_path, all_params=all_params, obs_p=obs_p)
    else:
        trainer.train(train_train_dataset, train_test_dataset, test_test_dataset,
            loss_weights, class_weighting, soft_label,
            num_epochs, batch_size, learning_rate, weight_decay,
            label_dir=label_dir, result_dir=os.path.join('result', naming, dataset_name,'split'+str(split)),
            log_freq=log_freq, log_train_results=log_train_results, args=args, all_params=all_params
        )
