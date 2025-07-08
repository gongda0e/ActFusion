import wandb
import os
from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import median_filter
import numpy as np
from tqdm import tqdm

from .dataset import restore_full_sequence
from .model.actfusion import ActFusion
from .utils import func_eval, get_labels_start_end_time, mode_filter, read_mapping_dict, eval_file, get_unique_list, get_unique_list_gt
from .vis import segment_bars

class Trainer:
    def __init__(self, encoder_params, decoder_params, diffusion_params,
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess, device, args=None):

        self.device = device
        self.num_classes = len(event_list)
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.event_list = event_list
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.postprocess = postprocess
        self.args = args

        self.model = ActFusion(encoder_params, decoder_params, diffusion_params, self.num_classes, self.device, args)
        wandb.watch(self.model)
        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))
        
        # Initialize best performance tracking
        self.best_tas_acc = 0.0
        self.best_lta_moc = 0.0
        self.best_both_score = 0.0
        self.best_tas_metrics = None
        self.best_lta_metrics = None
        self.best_combined_metrics = None

    def train(self, train_train_dataset, train_test_dataset, test_test_dataset, loss_weights, class_weighting, soft_label,
              num_epochs, batch_size, learning_rate, weight_decay, label_dir, result_dir, log_freq, log_train_results=True, args=None,
              all_params=None):

        device = self.device
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()

        restore_epoch = -1
        step = 1

        if os.path.exists(result_dir):
            if 'latest.pt' in os.listdir(result_dir):
                if os.path.getsize(os.path.join(result_dir, 'latest.pt')) > 0:
                    saved_state = torch.load(os.path.join(result_dir, 'latest.pt'))
                    self.model.load_state_dict(saved_state['model'])
                    optimizer.load_state_dict(saved_state['optimizer'])
                    restore_epoch = saved_state['epoch']
                    step = saved_state['step']

        if class_weighting:
            class_weights = train_train_dataset.get_class_weights()
            class_weights = torch.from_numpy(class_weights).float().to(device)
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights, reduction='none')
        else:
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        bce_criterion = nn.BCELoss(reduction='none')
        mse_criterion = nn.MSELoss(reduction='none')

        train_train_loader = torch.utils.data.DataLoader(
            train_train_dataset, batch_size=1, shuffle=True, num_workers=4)

        if result_dir:
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            logger = SummaryWriter(result_dir)

        for epoch in range(restore_epoch+1, num_epochs):
            self.model.train()
            epoch_running_loss = 0
            for _, data in enumerate(train_train_loader):
                feature, label, boundary, video = data
                feature, label, boundary = feature.to(device), label.to(device), boundary.to(device)

                loss_dict = self.model.get_training_loss(feature,
                    event_gt=F.one_hot(label.long(), num_classes=self.num_classes).permute(0, 2, 1),
                    boundary_gt=boundary,
                    encoder_ce_criterion=ce_criterion,
                    encoder_mse_criterion=mse_criterion,
                    encoder_boundary_criterion=bce_criterion,
                    decoder_ce_criterion=ce_criterion,
                    decoder_mse_criterion=mse_criterion,
                    decoder_boundary_criterion=bce_criterion,
                    soft_label=soft_label,
                    args=args
                )


                # ##############
                # # feature    torch.Size([1, F, T])
                # # label      torch.Size([1, T])
                # # boundary   torch.Size([1, 1, T])
                # # output    torch.Size([1, C, T])
                # ##################

                total_loss = 0

                for k,v in loss_dict.items():
                    total_loss += loss_weights[k] * v

                if result_dir:
                    for k,v in loss_dict.items():
                        logger.add_scalar(f'Train-{k}', loss_weights[k] * v.item() / batch_size, step)
                    logger.add_scalar('Train-Total', total_loss.item() / batch_size, step)

                total_loss /= batch_size
                total_loss.backward()

                epoch_running_loss += total_loss.item()

                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1

            epoch_running_loss /= len(train_train_dataset)
            wandb.log({"loss":epoch_running_loss})

            print(f'Epoch {epoch} - Running Loss {epoch_running_loss}')

            if result_dir:

                state = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step
                }

            if epoch % log_freq == 0:

                w_path = os.path.join(result_dir, 'log.txt')
                w = open(w_path, 'a')
                w_ = 'epoch'+str(epoch)+'\n'

                if result_dir:
                    torch.save(self.model.state_dict(), f'{result_dir}/epoch-{epoch}.model')
                    torch.save(state, f'{result_dir}/latest.pt')

                # for mode in ['encoder', 'decoder-noagg', 'decoder-agg']:
                for mode in ['decoder-agg']: # Default: decoder-agg. The results of decoder-noagg are similar
                    # Segmentation (TAS) inference
                    # TAS inference
                    test_result_dict = self.test(
                        test_test_dataset, mode, device, label_dir, args=args,
                        result_dir=result_dir, model_path=None, all_params=all_params, obs_p=1.0)
                    w_ = self._log_tas_results(test_result_dict, w_)

                    # LTA inference - obs_p 0.2
                    test_result_dict20 = self.test(
                        test_test_dataset, mode, device, label_dir, args=args,
                        result_dir=result_dir, model_path=None, all_params=all_params, obs_p=0.2)
                    w_ = self._log_lta_results(test_result_dict20, w_, obs_p=0.2)

                    # LTA inference - obs_p 0.3
                    test_result_dict30 = self.test(
                        test_test_dataset, mode, device, label_dir, args=args,
                        result_dir=result_dir, model_path=None, all_params=all_params, obs_p=0.3)
                    w_ = self._log_lta_results(test_result_dict30, w_, obs_p=0.3)

                    # === Best model saving ===
                    tas_f1s = [test_result_dict["F1@10"], test_result_dict["F1@25"], test_result_dict["F1@50"]]
                    tas_score = (test_result_dict["Acc"] + test_result_dict["Edit"] + sum(tas_f1s)) / (2 + len(tas_f1s))
                    lta_keys_02 = ["obs0.2_pred0.1", "obs0.2_pred0.2", "obs0.2_pred0.3", "obs0.2_pred0.5"]
                    lta_keys_03 = ["obs0.3_pred0.1", "obs0.3_pred0.2", "obs0.3_pred0.3", "obs0.3_pred0.5"]
                    lta_f1s_02 = [test_result_dict20.get(k, 0.0) for k in lta_keys_02]
                    lta_f1s_03 = [test_result_dict30.get(k, 0.0) for k in lta_keys_03]
                    lta_score = (sum(lta_f1s_02) + sum(lta_f1s_03)) / (len(lta_f1s_02) + len(lta_f1s_03))
                    self._save_best_models(tas_score, lta_score, result_dir, tas_metrics=test_result_dict, lta_metrics={'obs0.2': test_result_dict20, 'obs0.3': test_result_dict30})
                    # =====================

                    w.write(w_)
                    w.close()

                    if result_dir:
                        for k,v in test_result_dict.items():
                            logger.add_scalar(f'Test-{mode}-{k}', v, epoch)

                        np.save(os.path.join(result_dir,
                            f'test_results_{mode}_epoch{epoch}.npy'), test_result_dict)


                    if log_train_results:
                        train_result_dict = self.test(
                            train_test_dataset, mode, device, label_dir, args=args,
                            result_dir=result_dir, model_path=None)

                        if result_dir:
                            for k,v in train_result_dict.items():
                                logger.add_scalar(f'Train-{mode}-{k}', v, epoch)

                            np.save(os.path.join(result_dir,
                                f'train_results_{mode}_epoch{epoch}.npy'), train_result_dict)

                        for k,v in train_result_dict.items():
                            print(f'Epoch {epoch} - {mode}-Train-{k} {v}')

        if result_dir:
            logger.close()

        # Print best model performances
        final_summary = "\n" + "="*60 + "\n"
        final_summary += "BEST MODEL PERFORMANCES\n"
        final_summary += "="*60 + "\n"
        final_summary += f"Best TAS Score: {self.best_tas_acc:.2f}\n"
        final_summary += f"Best LTA Score: {self.best_lta_moc:.2f}\n"
        final_summary += f"Best Combined Score: {self.best_both_score:.2f}\n"
        
        if self.best_tas_metrics:
            final_summary += "\nBest TAS Metrics:\n"
            final_summary += f"  - Acc: {self.best_tas_metrics['Acc']:.2f}\n"
            final_summary += f"  - Edit: {self.best_tas_metrics['Edit']:.2f}\n"
            final_summary += f"  - F1@10: {self.best_tas_metrics['F1@10']:.2f}\n"
            final_summary += f"  - F1@25: {self.best_tas_metrics['F1@25']:.2f}\n"
            final_summary += f"  - F1@50: {self.best_tas_metrics['F1@50']:.2f}\n"
        
        if self.best_lta_metrics:
            final_summary += "\nBest LTA Metrics:\n"
            final_summary += "  obs_p=0.2:\n"
            for key in ["obs0.2_pred0.1", "obs0.2_pred0.2", "obs0.2_pred0.3", "obs0.2_pred0.5"]:
                if key in self.best_lta_metrics['obs0.2']:
                    final_summary += f"    - {key}: {self.best_lta_metrics['obs0.2'][key]:.2f}\n"
            final_summary += "  obs_p=0.3:\n"
            for key in ["obs0.3_pred0.1", "obs0.3_pred0.2", "obs0.3_pred0.3", "obs0.3_pred0.5"]:
                if key in self.best_lta_metrics['obs0.3']:
                    final_summary += f"    - {key}: {self.best_lta_metrics['obs0.3'][key]:.2f}\n"
        
        if self.best_combined_metrics:
            final_summary += "\nBest Combined Model Metrics:\n"
            final_summary += "  TAS Metrics:\n"
            tas_metrics = self.best_combined_metrics['tas']
            final_summary += f"    - Acc: {tas_metrics['Acc']:.2f}\n"
            final_summary += f"    - Edit: {tas_metrics['Edit']:.2f}\n"
            final_summary += f"    - F1@10: {tas_metrics['F1@10']:.2f}\n"
            final_summary += f"    - F1@25: {tas_metrics['F1@25']:.2f}\n"
            final_summary += f"    - F1@50: {tas_metrics['F1@50']:.2f}\n"
            final_summary += "  LTA Metrics:\n"
            lta_metrics = self.best_combined_metrics['lta']
            final_summary += "    obs_p=0.2:\n"
            for key in ["obs0.2_pred0.1", "obs0.2_pred0.2", "obs0.2_pred0.3", "obs0.2_pred0.5"]:
                if key in lta_metrics['obs0.2']:
                    final_summary += f"      - {key}: {lta_metrics['obs0.2'][key]:.2f}\n"
            final_summary += "    obs_p=0.3:\n"
            for key in ["obs0.3_pred0.1", "obs0.3_pred0.2", "obs0.3_pred0.3", "obs0.3_pred0.5"]:
                if key in lta_metrics['obs0.3']:
                    final_summary += f"      - {key}: {lta_metrics['obs0.3'][key]:.2f}\n"
        
        final_summary += "="*60 + "\n"
        
        # Print to console
        print(final_summary)
        
        # Write to log.txt file
        if result_dir:
            w_path = os.path.join(result_dir, 'log.txt')
            with open(w_path, 'a') as w:
                w.write(final_summary)



    def test_single_video(self, video_idx, test_dataset, mode, device, model_path=None, args=None, all_params=None, obs_p=0.2):
        assert(test_dataset.mode == 'test')
        assert(mode in ['encoder', 'decoder-noagg', 'decoder-agg'])
        assert(self.postprocess['type'] in ['median', 'mode', 'purge', None])

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        if self.set_sampling_seed:
            seed = video_idx
        else:
            seed = None

        with torch.no_grad():
            feature, label, _, video = test_dataset[video_idx]

            # feature:   [torch.Size([1, F, Sampled T])]
            # label:     torch.Size([1, Original T])
            # output: [torch.Size([1, C, Sampled T])]

            input_feats = feature

            # Check if this is anticipation mode (obs_p < 1.0)
            is_anticipation = obs_p < 1.0
            if is_anticipation:
                full_len = feature[0].size(-1)
                obs_len = round(full_len*obs_p)
                input_feats = [feature[i][:,:,:obs_len]
                        for i in range(len(feature))]

            if mode == 'encoder':
                output = [self.model.encoder(feature[i].to(device))
                       for i in range(len(feature))] # output is a list of tuples
                output = [F.softmax(i, 1).cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-agg':
                if is_anticipation:
                    output = [self.model.ddim_sample(input_feats[i].to(device), seed, full_len=full_len, args=args)
                            for i in range(len(input_feats))] # output is a list of tuples
                else:
                    output = [self.model.ddim_sample(feature[i].to(device), seed, args=args)
                            for i in range(len(feature))] # output is a list of tuples
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2
                output = [i.cpu() for i in output]

            if mode == 'decoder-noagg':  # temporal aug must be true
                output = [self.model.ddim_sample(feature[len(feature)//2].to(device), seed)] # output is a list of tuples
                output = [i.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = 0

            assert(output[0].shape[0] == 1)

            min_len = min([i.shape[2] for i in output])
            output = [i[:,:,:min_len] for i in output]
            output = torch.cat(output, 0)  # torch.Size([sample_rate, C, T])
            output = output.mean(0).numpy()

            if self.postprocess['type'] == 'median': # before restoring full sequence
                smoothed_output = np.zeros_like(output)
                for c in range(output.shape[0]):
                    smoothed_output[c] = median_filter(output[c], size=self.postprocess['value'])
                output = smoothed_output / smoothed_output.sum(0, keepdims=True)

            output = np.argmax(output, 0)


            output = restore_full_sequence(output,
                full_len=label.shape[-1],
                left_offset=left_offset,
                right_offset=right_offset,
                sample_rate=self.sample_rate
            )

            if self.postprocess['type'] == 'mode': # after restoring full sequence
                output = mode_filter(output, self.postprocess['value'])

            if self.postprocess['type'] == 'purge':

                trans, starts, ends = get_labels_start_end_time(output)

                for e in range(0, len(trans)):
                    duration = ends[e] - starts[e]
                    if duration <= self.postprocess['value']:

                        if e == 0:
                            output[starts[e]:ends[e]] = trans[e+1]
                        elif e == len(trans) - 1:
                            output[starts[e]:ends[e]] = trans[e-1]
                        else:
                            mid = starts[e] + duration // 2
                            output[starts[e]:mid] = trans[e-1]
                            output[mid:ends[e]] = trans[e+1]

            label = label.squeeze(0).cpu().numpy()
            assert(output.shape == label.shape)

            return video, output, label


    def test(self, test_dataset, mode, device, label_dir, result_dir=None, model_path=None, args=None, all_params=None, obs_p=0.2):

        assert(test_dataset.mode == 'test')

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        mapping_file = os.path.join('datasets', all_params['dataset_name'], 'mapping.txt')
        actions_dict = read_mapping_dict(mapping_file)
        actions_dict_inv = {v: k for k, v in actions_dict.items()}

        # Check if this is anticipation mode (obs_p < 1.0)
        is_anticipation = obs_p < 1.0
        if is_anticipation:
            eval_ps = [0.1, 0.2, 0.3, 0.5]
            T_actions = np.zeros((len(eval_ps), len(actions_dict)))
            F_actions = np.zeros((len(eval_ps), len(actions_dict)))

        with torch.no_grad():
            result_dict = {}
            for video_idx in tqdm(range(len(test_dataset))):
                video, pred, label = self.test_single_video(
                    video_idx, test_dataset, mode, device, model_path, args, all_params, obs_p=obs_p)

                pred_ant = pred
                pred = [self.event_list[int(i)] for i in pred]

                if not os.path.exists(os.path.join(result_dir, 'prediction')):
                    os.makedirs(os.path.join(result_dir, 'prediction'))

                file_name = os.path.join(result_dir, 'prediction', f'{video}.txt')
                file_ptr = open(file_name, 'w')
                file_ptr.write('### Frame level recognition: ###\n')
                file_ptr.write(' '.join(pred))
                file_ptr.close()

                if is_anticipation:
                    vis_path = os.path.join(result_dir, 'vis', 'ant'+str(obs_p), all_params['dataset_name'], 'split'+str(args.split))
                else:
                    vis_path = os.path.join(result_dir, 'vis', 'seg', all_params['dataset_name'], 'split'+str(args.split))
                if not os.path.exists(vis_path):
                    os.makedirs(vis_path)

                # Visualization code
                    gt_unique, gt_unique_len = get_unique_list_gt(label, actions_dict_inv)
                    pred_unique, pred_unique_len = get_unique_list(pred)
                    segment_bars(os.path.join(vis_path, f'{video}.png'), label, pred_ant, gt_unique=gt_unique, gt_unique_len=gt_unique_len,
                                  pred_unique=pred_unique, pred_unique_len=pred_unique_len)

                if is_anticipation:
                    total_len = len(label)
                    for i in range(len(eval_ps)):
                        eval_p=eval_ps[i]
                        eval_len = int((obs_p +eval_p)* total_len)
                        eval_pred = pred_ant[:eval_len]
                        T_action, F_action = eval_file(label, eval_pred, obs_p, actions_dict)
                        T_actions[i] += T_action
                        F_actions[i] += F_action

            if is_anticipation:
                for i in range(len(eval_ps)):
                    acc = 0
                    n = 0
                    for j in range(len(actions_dict)):
                        total_actions = T_actions + F_actions
                        if total_actions[i,j] != 0:
                            acc += float(T_actions[i,j]/total_actions[i,j])
                            n+=1

                    result = 'obs. %d '%int(100*obs_p) + 'pred. %d '%int(100*eval_ps[i])+'--> MoC: %.2f'%((float(acc)/n)*100)
                    result_dict['obs'+str(obs_p)+'_pred'+str(eval_ps[i])] = (float(acc)/n)*100  # Store as percentage
                    print(result)


        if is_anticipation:
            acc, edit, f1s = func_eval(
                label_dir, os.path.join(result_dir, 'prediction'), test_dataset.video_list, obs_p=obs_p)
        else:
            acc, edit, f1s = func_eval(
                label_dir, os.path.join(result_dir, 'prediction'), test_dataset.video_list, obs_p=1.0)

        result_dict['Acc'] = acc
        result_dict['Edit'] = edit
        result_dict['F1@10'] = f1s[0]
        result_dict['F1@25'] = f1s[1]
        result_dict['F1@50'] = f1s[2]

        if not is_anticipation:
            print("Acc:%.2f"%acc)
            print("Edit: %.2f"%edit)
            print('F1@10: %.2f'%f1s[0])
            print('F1@25: %.2f'%f1s[1])
            print('F1@50: %.2f'%f1s[2])

        return result_dict

    def _log_tas_results(self, result_dict, text_buffer):
        """Log TAS (Temporal Action Segmentation) results to wandb and text buffer"""
        # Log to wandb
        wandb.log({"acc": result_dict["Acc"]})
        wandb.log({"edit": result_dict["Edit"]})
        wandb.log({"F1@10": result_dict["F1@10"]})
        wandb.log({"F1@25": result_dict["F1@25"]})
        wandb.log({"F1@50": result_dict["F1@50"]})
        
        # Add to text buffer
        text_buffer += "TAS Results:\n"
        text_buffer += "Acc: %.2f" % result_dict["Acc"] + '\n'
        text_buffer += "Edit: %.2f" % result_dict["Edit"] + '\n'
        text_buffer += "F1@10: %.2f" % result_dict["F1@10"] + '\n'
        text_buffer += "F1@25: %.2f" % result_dict["F1@25"] + '\n'
        text_buffer += "F1@50: %.2f" % result_dict["F1@50"] + '\n\n'
        
        return text_buffer

    def _log_lta_results(self, result_dict, text_buffer, obs_p):
        """Log LTA (Long-Term Anticipation) results to wandb and text buffer"""
        # Log to wandb
        wandb.log({f"obs{obs_p}_pred0.1": result_dict[f"obs{obs_p}_pred0.1"]})
        wandb.log({f"obs{obs_p}_pred0.2": result_dict[f"obs{obs_p}_pred0.2"]})
        wandb.log({f"obs{obs_p}_pred0.3": result_dict[f"obs{obs_p}_pred0.3"]})
        wandb.log({f"obs{obs_p}_pred0.5": result_dict[f"obs{obs_p}_pred0.5"]})
        
        # Add to text buffer
        text_buffer += f"LTA Results (obs_p={obs_p}):\n"
        text_buffer += f"obs{obs_p}_pred0.1: %.2f" % result_dict[f"obs{obs_p}_pred0.1"] + '\n'
        text_buffer += f"obs{obs_p}_pred0.2: %.2f" % result_dict[f"obs{obs_p}_pred0.2"] + '\n'
        text_buffer += f"obs{obs_p}_pred0.3: %.2f" % result_dict[f"obs{obs_p}_pred0.3"] + '\n'
        text_buffer += f"obs{obs_p}_pred0.5: %.2f" % result_dict[f"obs{obs_p}_pred0.5"] + '\n\n'
        
        return text_buffer

    def _save_best_models(self, tas_acc, lta_moc, result_dir, tas_metrics=None, lta_metrics=None):
        """Save best models based on TAS accuracy, LTA MoC, and combined score"""
        # LTA metric is already scaled to percentage (0.1-0.3 → 10-30)
        # Calculate combined score with equal weights
        combined_score = (tas_acc + lta_moc) / 2
        
        # Check and save best TAS model
        if tas_acc > self.best_tas_acc:
            self.best_tas_acc = tas_acc
            self.best_tas_metrics = tas_metrics
            torch.save(self.model.state_dict(), f'{result_dir}/best_tas_model.pth')
            print(f'New best TAS model saved! Score: {tas_acc:.2f}')
            if tas_metrics:
                print(f'  TAS Metrics - Acc: {tas_metrics["Acc"]:.2f}, Edit: {tas_metrics["Edit"]:.2f}, F1@10: {tas_metrics["F1@10"]:.2f}, F1@25: {tas_metrics["F1@25"]:.2f}, F1@50: {tas_metrics["F1@50"]:.2f}')
        
        # Check and save best LTA model
        if lta_moc > self.best_lta_moc:
            self.best_lta_moc = lta_moc
            self.best_lta_metrics = lta_metrics
            torch.save(self.model.state_dict(), f'{result_dir}/best_lta_model.pth')
            print(f'New best LTA model saved! Score: {lta_moc:.2f}')
            if lta_metrics:
                print(f'  LTA Metrics - obs0.2: {lta_metrics["obs0.2"]["obs0.2_pred0.1"]:.2f}/{lta_metrics["obs0.2"]["obs0.2_pred0.2"]:.2f}/{lta_metrics["obs0.2"]["obs0.2_pred0.3"]:.2f}/{lta_metrics["obs0.2"]["obs0.2_pred0.5"]:.2f}, obs0.3: {lta_metrics["obs0.3"]["obs0.3_pred0.1"]:.2f}/{lta_metrics["obs0.3"]["obs0.3_pred0.2"]:.2f}/{lta_metrics["obs0.3"]["obs0.3_pred0.3"]:.2f}/{lta_metrics["obs0.3"]["obs0.3_pred0.5"]:.2f}')
        
        # Check and save best combined model
        if combined_score > self.best_both_score:
            self.best_both_score = combined_score
            self.best_combined_metrics = {'tas': tas_metrics, 'lta': lta_metrics}
            torch.save(self.model.state_dict(), f'{result_dir}/best_combined_model.pth')
            print(f'New best combined model saved! Score: {combined_score:.2f} (TAS: {tas_acc:.2f}, LTA: {lta_moc:.2f})')
            if tas_metrics and lta_metrics:
                print(f'  TAS Metrics - Acc: {tas_metrics["Acc"]:.2f}, Edit: {tas_metrics["Edit"]:.2f}, F1@10: {tas_metrics["F1@10"]:.2f}, F1@25: {tas_metrics["F1@25"]:.2f}, F1@50: {tas_metrics["F1@50"]:.2f}')
                print(f'  LTA Metrics - obs0.2: {lta_metrics["obs0.2"]["obs0.2_pred0.1"]:.2f}/{lta_metrics["obs0.2"]["obs0.2_pred0.2"]:.2f}/{lta_metrics["obs0.2"]["obs0.2_pred0.3"]:.2f}/{lta_metrics["obs0.2"]["obs0.2_pred0.5"]:.2f}, obs0.3: {lta_metrics["obs0.3"]["obs0.3_pred0.1"]:.2f}/{lta_metrics["obs0.3"]["obs0.3_pred0.2"]:.2f}/{lta_metrics["obs0.3"]["obs0.3_pred0.3"]:.2f}/{lta_metrics["obs0.3"]["obs0.3_pred0.5"]:.2f}')