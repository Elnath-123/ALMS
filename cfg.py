##### config #####

seed = 27509

_C_ucf = {
    'dataset_dir': './ucf/ucf_dataset/partition/ucf.npy',
    'save_model_dir': './ucf/ucf_model/',
    'train_size': 1600 // 2,
    'test_size': 1600 // 2,
    'feature_in': 256,
    'latent_1': 350,
    'latent_2': 450,
    'output_encoder': 512,
    'sketch_size': 128,
    'class_num': 11,
    'SNR': 5,
    'batch_size' : 2414 // 8 * 3
}

_C_musk = {
    'dataset_dir': './musk/musk_dataset/partition/musk.npy',
    'save_model_dir': './musk/musk_model/',
    'train_size': 6598 // 2,
    'test_size': 6598 // 2,
    'feature_in': 166,
    'latent_1': 180,
    'latent_2': 220,
    'output_encoder': 256,
    'sketch_size': 250,
    'class_num': 2,
    'SNR': 5,
    'batch_size' : 6598 // 8 * 3
}

_C_ESR = {
    'dataset_dir': './ESR/ESR_dataset/partition/ESR.npy',
    'save_model_dir': './ESR/ESR_model/',
    'train_size': 11500 // 2,
    'test_size': 11500 // 2,
    'feature_in': 178,
    'latent_1': 170,
    'latent_2': 165,
    'output_encoder': 160,
    'sketch_size': 160,
    'class_num': 5,
    'SNR': 5,
    'batch_size' : 11500 // 8 * 3
}

_C_protein = {
    'dataset_dir': './protein/protein_dataset/partition/protein.npy',
    'save_model_dir': './protein/protein_model/',
    'train_size': 1080 // 2,
    'test_size': 1080 // 2,
    'feature_in': 77,
    'latent_1': 73,
    'latent_2': 70,
    'output_encoder': 60,
    'sketch_size': 60,
    'class_num': 8,
    'SNR': 5,
    'batch_size' : 5822 // 8 * 3
}


_C_swarm = {
    'dataset_dir': './swarm/swarm_dataset/partition/swarm.npy',
    'save_model_dir': './swarm/swarm_model/',
    'train_size': 24016 // 2,
    'test_size': 24016 // 2,
    'feature_in': 2400,
    'latent_1': 2000,
    'latent_2': 1500,
    'output_encoder': 1000,
    'sketch_size': 500,
    'class_num': 2,
    'SNR': 5,
    'batch_size' : 24016 // 8 * 3
}

_C_mediamill = {
    'dataset_dir': './mediamill/mediamill_dataset/partition/mediamill.npy',
    'save_model_dir': './mediamill/mediamill_model/',
    'train_size': 43907 // 2,
    'test_size': 43907 // 2,
    'feature_in': 120,
    'latent_1': 160,
    'latent_2': 210,
    'output_encoder': 256,
    'sketch_size': 250,
    'class_num': 2,
    'SNR': 20,
    'batch_size' : 43907 // 8 * 3
}
def print_config(_C):
    print('---------- configuration information ----------')
    for k, v in _C.items():
        print('{}: {}'.format(k, v))
    print('----------------------end----------------------')