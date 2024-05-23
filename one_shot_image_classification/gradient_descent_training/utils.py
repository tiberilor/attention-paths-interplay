import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np

from autoaugment import CIFAR10Policy, SVHNPolicy
from criterions import LabelSmoothingCrossEntropyLoss
from da import RandomCropPaste

def get_criterion(args):
    if args.criterion=="ce":
        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropyLoss(args.num_classes, smoothing=args.smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
    elif args.criterion in ["mse", "sigm_mse", "binary_mse"]:
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"{args.criterion}?")

    return criterion

def get_model(args):
    if args.model_name == 'my_vit':
        from my_vit import MyViT
        net = MyViT(
            channels = args.in_c, 
            num_classes = args.num_classes, 
            patch_size = args.patch_size,
            image_size = args.size, 
            mlp_dim=args.d_ff,
            depth = args.num_layers,
            dim_head = args.dim_head,
            dim = args.d_model,
            heads=args.num_heads,
            pool=args.readout_type,
            use_sin_pos_enc=args.use_sin_pos_enc,
            use_random_first_projection=args.use_random_first_projection,
            freeze_icl_label_embedding=args.freeze_icl_label_embedding,
            in_context_learning=args.in_context_learning,
            icl_num_shots=args.icl_num_shots,
            no_feedforward=args.vit_no_feedforward,
            )
    elif args.model_name == 'my_dual_head_no_learning_input_constrained_linear_vit':
        from my_vit import DualHeadNoLearningInputConstrainedLinearViT
        net = DualHeadNoLearningInputConstrainedLinearViT(
            channels = args.in_c,
            num_classes = args.num_classes, 
            patch_size = args.patch_size,
            image_size = args.size,
            num_layers = args.num_layers,
            dim_head = args.dim_head,
            qk_dim_head = args.qk_dim_head,
            d_model = args.d_model,
            num_heads = args.num_heads,
            num_sum_heads = args.num_sum_heads,
            pool=args.readout_type,
            concat_pos_emb_dim=args.concat_pos_emb_dim,
            use_random_first_projection=args.use_random_first_projection,
            use_sin_pos_enc=args.use_sin_pos_enc,
            no_cls_token=args.no_cls_token,
            no_residual=args.no_residual,
            concat_pos_enc=args.concat_pos_enc,
            use_random_position_encoding=args.use_random_position_encoding,
            remove_diag_scale=args.remove_diag_scale,
            use_parallel=args.use_parallel,
            learn_attention=args.learn_attention,
            add_learned_input_layer=args.add_learned_input_layer,
            remove_diagonal_init=args.remove_diagonal_init,
            remove_nonlinear_input_projection=args.remove_nonlinear_input_projection,
            in_context_learning=args.in_context_learning,
            icl_num_shots=args.icl_num_shots,
            additive_icl_label_embedding=args.additive_icl_label_embedding,
            freeze_icl_label_embedding=args.freeze_icl_label_embedding,
            readout_column_index=args.readout_column_index,
            freeze_value=args.freeze_value,
            freeze_input_projection=args.freeze_input_projection,
            freeze_readout=args.freeze_readout,
            add_biases=args.add_biases,
            )
    elif args.model_name == 'my_dual_head_no_learning_input_constrained_linear_text_trafo':
        from my_vit import DualHeadNoLearningInputConstrainedLinearTextTrafo
        net = DualHeadNoLearningInputConstrainedLinearTextTrafo(
            input_vocab_size = args.input_vocab_size,  # new
            num_classes = args.num_classes, 
            max_seq_len = args.max_seq_length,  # new
            token_embedding_dim = args.token_embedding_dim,  # new
            num_layers = args.num_layers,
            dim_head = args.dim_head,
            qk_dim_head = args.qk_dim_head,
            d_model = args.d_model,
            num_heads = args.num_heads,
            num_sum_heads = args.num_sum_heads,
            pool=args.readout_type,
            concat_pos_emb_dim=args.concat_pos_emb_dim,
            use_sin_pos_enc=args.use_sin_pos_enc,
            no_cls_token=args.no_cls_token,
            no_residual=args.no_residual,
            concat_pos_enc=args.concat_pos_enc,
            use_random_position_encoding=args.use_random_position_encoding,
            remove_diag_scale=args.remove_diag_scale,
            learn_attention=args.learn_attention,
            add_learned_input_layer=args.add_learned_input_layer,
            remove_diagonal_init=args.remove_diagonal_init,
            remove_nonlinear_input_projection=args.remove_nonlinear_input_projection,
            readout_column_index=args.readout_column_index,
            freeze_value=args.freeze_value,
            freeze_input_projection=args.freeze_input_projection,
            freeze_readout=args.freeze_readout,
            add_biases=args.add_biases,
            dropout=args.dropout,
            )
    else:
        raise NotImplementedError(f"{args.model_name} is not implemented yet...")

    args.loginf(net)
    args.loginf(f'Number of parameters: {net.num_params()}')
    return net

def get_transform(args):
    train_transform = []
    test_transform = []
    train_transform += [
        transforms.RandomCrop(size=args.size, padding=args.padding)
    ]
    if args.dataset != 'svhn':
        train_transform += [transforms.RandomHorizontalFlip()]
    
    if args.autoaugment:
        if args.dataset == 'c10' or args.dataset=='c100':
            train_transform.append(CIFAR10Policy())
        elif args.dataset == 'svhn':
            train_transform.append(SVHNPolicy())
        else:
            print(f"No AutoAugment for {args.dataset}")   

    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]
    if args.rcpaste:
        train_transform += [RandomCropPaste(size=args.size)]
    
    test_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform
    

def get_dataset(args):
    root = "data"
    if args.dataset == "c10":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]

        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR10(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR10(root, train=False, transform=test_transform, download=True)

    elif args.dataset == "c100":
        args.in_c = 3
        args.num_classes=100
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR100(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR100(root, train=False, transform=test_transform, download=True)

    elif args.dataset == "svhn":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.SVHN(root, split="train",transform=train_transform, download=True)
        test_ds = torchvision.datasets.SVHN(root, split="test", transform=test_transform, download=True)

    else:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")
    
    return train_ds, test_ds

def get_experiment_name(args):
    experiment_name = f"{args.model_name}_{args.dataset}"
    if args.autoaugment:
        experiment_name+="_aa"
    if args.label_smoothing:
        experiment_name+="_ls"
    if args.rcpaste:
        experiment_name+="_rc"
    if args.cutmix:
        experiment_name+="_cm"
    if args.mixup:
        experiment_name+="_mu"
    if args.no_cls_token:
        experiment_name+="_gap"
    print(f"Experiment:{experiment_name}")
    return experiment_name
