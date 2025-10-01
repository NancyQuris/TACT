
dataset_defaults = {
    'birdcalls': {
        'epochs':  100, 
        'batch_size': 16,
        'split_scheme': 'combined',
        'model': 'vitb32',
        'optimiser': 'AdamW',
        'optimiser_args': {
            'lr': 5e-5,
            'weight_decay': 0.001,
        },
        'selection_metric': 'F1-macro_all', 
        'groupby_fields': ['location'],
        'n_groups_per_batch': 2,
        'scheduler': None,
        'print_iters': 2000,
        'group_num': None,
        'group_by_label': False,  
    },
    'camelyon': {
        'epochs': 30,
        'batch_size': 32,
        'model': 'vitb32',
        'optimiser': 'SGD',
        'optimiser_args': {
            'momentum': 0.9,
            'lr': 5e-5,
            'weight_decay': 0,
        },
        'selection_metric': 'acc_avg',
        'print_iters': 2000,
        'group_num': None,
        'group_by_label': False,  
    },
    'civil': {
        'epochs': 5,
        'batch_size': 8,
        'optimiser': 'Adam',
        'reweight_groups': False,
        'model': 'distilbert',
        'optimiser_args': {
            'lr': 1e-5,
            'weight_decay': 0.0,
        },
        'selection_metric': 'acc_wg',
        'group_num': None,
        'group_by_label': False,  
    },
}

