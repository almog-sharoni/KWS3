# Configuration settings for datasets, data loaders, and other related parameters.

dataset = {
    'speech_commands': {
        'path': '/path/to/speech_commands',
        'num_classes': 12,
        'sample_rate': 16000,
        'duration': 1.0,
        'num_samples': 16000,
    },
    'background_noise': {
        'path': '/path/to/background_noise',
        'sample_rate': 16000,
    }
}

data_loader = {
    'batch_size': 32,
    'num_workers': 4,
    'pin_memory': True,
    'drop_last': True,
}

model = {
    'num_mamba_layers': 2,
    'dropout_rate': 0.5,
    'use_cls_token': True,
}

optimizer = {
    'lr': 0.001,
    'weight_decay': 1e-5,
    'lookahead': {
        'steps': 5,
        'alpha': 0.5,
    }
}

scheduler = {
    'reduce_lr_on_plateau': {
        'factor': 0.1,
        'patience': 5,
        'verbose': True,
    }
}

training = {
    'num_epochs': 100,
    'early_stopping': True,
    'patience': 10,
}