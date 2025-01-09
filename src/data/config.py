configs = {'d_state': 51, 'd_conv': 10, 'expand': 2, 'batch_size': 128, 'dropout_rate': 0.134439213335519, 'num_mamba_layers': 2, 'n_mfcc': 23, 'n_fft': 475, 'hop_length': 119, 'n_mels': 61, 'noise_level': 0.2582577623788829, 'lr': 0.0011942156978344588, 'weight_decay': 2.5617519345807027e-05}

dataset = {
    'fixed_length': 16000,
    'n_mfcc': configs['n_mfcc'],  # Use from configs
    'n_fft': configs['n_fft'],    # Use from configs
    'hop_length': configs['hop_length'],  # Use from configs
    'n_mels': configs['n_mels'],  # Use from configs
    'noise_level': configs['noise_level']  # Use from configs
}

data_loader = {
    'batch_size': configs['batch_size'],  # Use from configs
    'num_workers': 4,
    'prefetch_factor': 2
}
# input_dim=n_mfcc*3, 
# d_model= (16000 // hop_length) + 1 + 1  ,
model = {

    'input_dim': configs['n_mfcc'] * 3,  # Use from configs
    'd_model': (dataset['fixed_length'] // dataset['hop_length']) + 1 + 1,  # Use from configs
    'd_state': configs['d_state'],  # Use from configs
    'd_conv': configs['d_conv'],    # Use from configs
    'expand': configs['expand'],    # Use from configs
    'num_mamba_layers': configs['num_mamba_layers'],  # Use from configs
    'dropout_rate': configs['dropout_rate'],  # Use from configs
    # 'label_names': ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
    'label_names': ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes','silence','unknown']
}

optimizer = {
    'lr': configs['lr'],  # Use from configs
    'weight_decay': configs['weight_decay'],  # Use from configs
    'lookahead': {
        'k': 5,
        'alpha': 0.5
    }
}

scheduler = {
    'reduce_lr_on_plateau': {
        'mode': 'min',
        'factor': 0.1,
        'patience': 3
    }
}

training = {
    'num_epochs': 100
}

# config.py

# dataset = {
#     'fixed_length': 16000,
#     'n_mfcc': 13,
#     'n_fft': 640,
#     'hop_length': 320,
#     'n_mels': 40,
#     'noise_level': 0.05
# }

# data_loader = {
#     'batch_size': 32,
#     'num_workers': 4,
#     'prefetch_factor': 2
# }

# model = {
#     'input_dim': 39,
#     'd_model': 52,
#     'd_state': 16,
#     'd_conv': 4,
#     'expand': 2,
#     'num_mamba_layers': 1,
#     'dropout_rate': 0.1,
#     'label_names': ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
# }

# optimizer = {
#     'lr': 0.0024,
#     'weight_decay': 2.80475e-05,
#     'lookahead': {
#         'k': 5,
#         'alpha': 0.5
#     }
# }

# scheduler = {
#     'reduce_lr_on_plateau': {
#         'mode': 'min',
#         'factor': 0.1,
#         'patience': 3
#     }
# }

# training = {
#     'num_epochs': 100
# }

#  {'d_state': 54, 'd_conv': 2, 'expand': 4, 'batch_size': 48, 'dropout_rate': 0.5485717021018514, 'num_mamba_layers': 2, 'n_mfcc': 18, 'n_fft': 604, 'hop_length': 221, 'n_mels': 67, 'noise_level': 0.21777823908139465, 'lr': 0.0015442563359956897, 'weight_decay': 3.7694282507597687e-06}
# dataset = {
#     'fixed_length': 16000,
#     'n_mfcc': 13,
#     'n_fft': 640,
#     'hop_length': 320,
#     'n_mels': 40,
#     'noise_level': 0.05
# }

# data_loader = {
#     'batch_size': 48,
#     'num_workers': 4,
#     'prefetch_factor': 2
# }

# model = {
#     'input_dim': 39,
#     'd_model': 52,
#     'd_state': 54,
#     'd_conv': 2,
#     'expand': 4,
#     'num_mamba_layers': 2,
#     'dropout_rate': 0.5485717021018514,
#     'label_names': ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
# }

# optimizer = {
#     'lr': 0.0015442563359956897,
#     'weight_decay': 3.7694282507597687e-06,
#     'lookahead': {
#         'k': 5,
#         'alpha': 0.5
#     }
# }

# scheduler = {
#     'reduce_lr_on_plateau': {
#         'mode': 'min',
#         'factor': 0.1,
#         'patience': 3
#     }
# }

# training = {
#     'num_epochs': 100
# }

# configs = {'d_state': 56, 'd_conv': 4, 'expand': 4, 'batch_size': 52, 'dropout_rate': 0.6986202181932293, 'num_mamba_layers': 1, 'n_mfcc': 22, 'n_fft': 704, 'hop_length': 247, 'n_mels': 41, 'noise_level': 0.17008641975092453, 'lr': 0.004762135680035422, 'weight_decay': 9.205584198870273e-06}
# configs = {'d_state': 64, 'd_conv': 2, 'expand': 4, 'batch_size': 43, 'dropout_rate': 0.602554749852884, 'num_mamba_layers': 2, 'n_mfcc': 20, 'n_fft': 670, 'hop_length': 254, 'n_mels': 82, 'noise_level': 0.28871524345963123, 'lr': 0.0013258735838467604, 'weight_decay': 4.868052802370925e-06}
