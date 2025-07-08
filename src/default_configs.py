import os
import json
import copy

params_50salads = {
   "naming":"50salads",
   "root_data_dir":"./datasets",
   "dataset_name":"50salads",
   "sample_rate":8,
   "temporal_aug":True,
   "pos":"rel",
   "n_mask":10,
   "patch_size":10,
   "encoder_params":{
      "use_instance_norm":False,
      "num_layers":10,
      "num_f_maps":64,
      "input_dim":2048,
      "kernel_size":5,
      "normal_dropout_rate":0.5,
      "channel_dropout_rate":0.5,
      "temporal_dropout_rate":0.5,
      "feature_layer_indices":[
         5,
         7,
         9
      ]
   },
   "decoder_params":{
      "num_layers":8,
      "num_f_maps":24,
      "time_emb_dim":512,
      "kernel_size":7,
      "dropout_rate":0.1,
   },
   "diffusion_params":{
      "timesteps":1000,
      "sampling_timesteps":25,
      "ddim_sampling_eta":1.0,
      "snr_scale":1.0,
      "cond_types":[
         "full",
         "ant",
         "random_num_patch",
         "boundary05-",
         "segment=2",
         "segment=2"
      ],
     "detach_decoder": False,
   },
   "loss_weights":{
      "encoder_ce_loss":0.5,
      "encoder_mse_loss":0.1,
      "encoder_boundary_loss":0.0,
      "decoder_ce_loss":0.5,
      "decoder_mse_loss":0.1,
      "decoder_boundary_loss":0.1
   },
   "batch_size":4,
   "learning_rate":0.0005,
   "weight_decay":0,
   "num_epochs":7001,
   "log_freq":100,
   "class_weighting":True,
   "set_sampling_seed":True,
   "boundary_smooth":20,
   "soft_label": None,
   "log_train_results":False,
   "postprocess":{
      "type":"median",
      "value":30
   },
}

params_breakfast = {
   "naming":"Breakfast",
   "root_data_dir":"./datasets",
   "dataset_name":"breakfast",
   "sample_rate":1,
   "temporal_aug":True,
   "pos":"rel",
   "n_mask":10,
   "patch_size":10,
   "encoder_params":{
      "use_instance_norm":False,
      "num_layers":12,
      "num_f_maps":256,
      "input_dim":2048,
      "kernel_size":5,
      "normal_dropout_rate":0.5,
      "channel_dropout_rate":0.1,
      "temporal_dropout_rate":0.1,
      "feature_layer_indices":[
         7,
         8,
         9
      ]
   },
   "decoder_params":{
      "num_layers":8,
      "num_f_maps":128,
      "time_emb_dim":512,
      "kernel_size":5,
      "dropout_rate":0.1
   },
   "diffusion_params":{
      "timesteps":1000,
      "sampling_timesteps":25,
      "ddim_sampling_eta":1.0,
      "snr_scale":0.5,
      "cond_types":[
         "ant",
         "full",
         "boundary03-",
         "segment=1",
         "segment=1",
         "random_patch"
      ],
      "detach_decoder":False,
   },
   "loss_weights":{
      "encoder_ce_loss":0.5,
      "encoder_mse_loss":0.025,
      "encoder_boundary_loss":0.0,
      "decoder_ce_loss":0.5,
      "decoder_mse_loss":0.025,
      "decoder_boundary_loss":0.1
   },
   "batch_size":16,
   "learning_rate":0.0001,
   "weight_decay":0,
   "num_epochs":1001,
   "log_freq":10,
   "class_weighting":True,
   "set_sampling_seed":True,
   "boundary_smooth":3,
   "soft_label":4,
   "log_train_results":False,
   "postprocess":{
      "type":"median",
      "value":15
   },
}

if __name__ == "__main__":
    params = copy.deepcopy(params_50salads)
    params['naming'] = '50salads'

    if not os.path.exists('configs'):
        os.makedirs('configs')

    file_name = os.path.join('configs', f'{params["naming"]}.json')
    with open(file_name, 'w') as outfile:
        json.dump(params, outfile, ensure_ascii=False)
    print(f"Saved: {file_name}")

    ###################### Breakfast #######################
    params = copy.deepcopy(params_breakfast)
    params['naming'] = 'Breakfast'

    if not os.path.exists('configs'):
        os.makedirs('configs')

    file_name = os.path.join('configs', f'{params["naming"]}.json')
    with open(file_name, 'w') as outfile:
        json.dump(params, outfile, ensure_ascii=False)
    print(f"Saved: {file_name}")
