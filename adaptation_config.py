def get_adapt_hparams_list(args):
    algo_name = args.algorithm
    if algo_name in ['TACT']:
        adapt_hparams_dict = {
            'num_aug': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            'start_pc': [0],
            'num_pcs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        }
    elif algo_name in ['TACT_adapt']:
        adapt_hparams_dict = {
            'num_aug': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            'start_pc': [0],
            'num_pcs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            'adaptation_lr': [1e-3, 1e-4, 1e-5, 1e-6],
            'entropy_weighting': [0.1, 0.5, 1, 5, 10, 50, 100, 500]
        }
    elif algo_name == 'T3A': 
        adapt_hparams_dict = {
            'filter_K': [1, 5, 20, 50, 100, -1], 
        }
    elif algo_name == 'PASLE':
        adapt_hparams_dict = {
            'adaptation_lr': [1e-3, 1e-4, 1e-5, 1e-6], 
            'param_to_adapt': ['affine', 'head', 'body', 'all'],
            'optimizer': ['Adam'],
            'thresh': [0.2, 0.4, 0.6, 0.8],
            'thresh_gap': [0.1],
            'thresh_des': [1e-3, 1e-4],
            'temp': [3],
            'buffer_size': [16],
        }
    elif algo_name in ['DeYO']:
        adapt_hparams_dict = {
            'adaptation_lr': [1e-3, 1e-4, 1e-5, 1e-6], 
            'adaption_step': [1],  
            'episodic': [False],
            'adaptation_momentum': [0.9],  
            'patch_len': [4],
            'aug_type': ['patch', 'pixel', 'occ'], 
            'deyo_margin': [0.5],
            'margin_e0': [0.4], 
            'plpd_threshold': [0.2, 0.3, 0.5], 
            'filter_ent': [1],
            'filter_plpd': [1],
            'reweight_ent': [1],
            'reweight_plpd': [1]
        }
        # occulsion size = original image size/2, occulude the center 
        if args.dataset in ['imagenet_r', 'imagenet_sketch', 'imagenetv2', 'birdcalls']:
            adapt_hparams_dict['occlusion_size'] = [112]
            adapt_hparams_dict['row_start'] = [56]
            adapt_hparams_dict['column_start'] = [56]
        elif args.dataset in ['camelyon']:
            adapt_hparams_dict['occlusion_size'] = [48]
            adapt_hparams_dict['row_start'] = [28]
            adapt_hparams_dict['column_start'] = [28]
    elif algo_name in 'SAR':
        adapt_hparams_dict = {
            'adaptation_lr': [1e-3, 1e-4, 1e-5, 1e-6], 
            'adaption_step': [1], 
            'episodic': [False],
            'adaptation_momentum': [0.9],
            'margin_e0': [0.4], 
            'reset_constant_em': [0.2]
        }
    elif algo_name in ['Tent']:  
        adapt_hparams_dict = {
            'adaptation_lr': [1e-3, 1e-4, 1e-5, 1e-6], 
            'adaptation_step': [1], 
            'episodic': [False],
            'adaptation_momentum': [0.9],
            'optimizer': ['SGD']
        }
    elif algo_name == 'TSD':
        adapt_hparams_dict = {
            'adaptation_lr': [1e-3, 1e-4, 1e-5, 1e-6], 
            'param_to_adapt': ['affine', 'all', 'head', 'body', ],
            'adaptation_step': [1],
            'episodic': [False],
            'filter_K': [1, 5, 20, 50, 100, -1],  
            'lam': [0.1]
        }
    elif algo_name == 'TAST':
        adapt_hparams_dict = {
            'adaptation_lr': [1e-3, 1e-4, 1e-5, 1e-6], 
            'adaptation_step': [1, 3],
            'num_support': [1, 2, 4, 8],
            'filter_K': [1, 5, 20, 50, 100, -1], 
            'num_ensemble': [20] 
        }
    elif algo_name == 'LAME':
        adapt_hparams_dict = {
            'knn': [1,3,5],
            'affinity': ['kNN', 'rbf', 'linear'],
            'sigma': [1],
            'force_symmetry': [True]
        }
    elif algo_name == 'FOA':
        adapt_hparams_dict = {
            'num_prompts': [3],
            'fitness_lambda': [0.4, 0.2]
        }
    elif algo_name in ['SHOT']:  
        adapt_hparams_dict = {
            'adaptation_lr': [1e-3, 1e-4, 1e-5, 1e-6], 
            'beta': [0.9],
            'theta': [0.1],
            'adaptation_step': [1], 
            'episodic': [False],
        }
    else:
        adapt_hparams_dict = {}
    
    adapt_hparams_dict['lr'] = [args.optimiser_args['lr']]
    adapt_hparams_dict['weight_decay'] = [args.optimiser_args['weight_decay']]

    return adapt_hparams_dict