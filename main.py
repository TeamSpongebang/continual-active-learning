################################################################################
# Date: 05-17-2022                                                             #
# Author: Jia Shi, Zhiqiu Lin                                                  #
# E-mail: jiashi@andrew.cmu.edu, zl279@cornell.edu                             #
# Website: https://clear-benchmark.github.io                                   #
################################################################################

################################################################################
# Date: 07-28-2023                                                             #
# Author: Jeongkyun Park                                                       #
# E-mail: park32323@gmail.com                                                  #
# Expand CLEAR CVPR challenge codes to active learning.                        #
################################################################################


import os
import json
import argparse
from pathlib import Path
import copy
from datetime import datetime

import numpy as np
import torch
import torchvision
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    timing_metrics,
    cpu_usage_metrics,
    confusion_matrix_metrics,
    disk_usage_metrics,
)
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.training.supervised import Naive
from avalanche.benchmarks.classic.clear import CLEAR, CLEARMetric

from arguments import add_training_args, add_query_args
from al import ActivePool
from al.methods import NAME_TO_CLS
from al.methods.random import RandomSampling
from trainer.ensemble_trainer import ensemble_trainer
from trainer.trainer import default_trainer
from utils import set_seed, write_json


def make_scheduler(optimizer, scheduler_args):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_args)
    return scheduler

def get_model(args):
    if args.arch == 'resnet18':
        return torchvision.models.resnet18(weights="IMAGENET1K_V1")

    elif args.arch == 'resnet50':
        return torchvision.models.resnet50(weights="IMAGENET1K_V1")

    elif args.arch == 'vit16':
        return torchvision.models.vit_b_16(weights="IMAGENET1K_V1")

    elif args.arch == 'vit32':
        return torchvision.models.vit_b_32(weights="IMAGENET1K_V1")

    else:
        raise NotImplementedError(f"Architecture {args.arch} not supported.")

def get_optimizer_schedular(model, optim_args, scheduler_args):
    optimizer = torch.optim.SGD(model.parameters(), **optim_args)
    scheduler = make_scheduler(optimizer, scheduler_args)
    return optimizer, scheduler

def get_cumulative_dataset(args, train_stream, index):
    data_set = torch.utils.data.ConcatDataset(
        [train_stream[i].dataset.train() for i in range(index+1)])
    print('length of dataset: ')
    print(len(data_set))
    train_loader = torch.utils.data.DataLoader(data_set,
                            batch_size=args.batch_size, shuffle=True)
    return train_loader

def get_dataset(train_stream, index):
    data_set = train_stream[index].dataset.train()
    return data_set

class QDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, return_fn): 
        self.dataset = dataset
        self.return_fn = return_fn
    def __len__(self): 
        return len(self.dataset)
    def __getitem__(self, index):
        return self.return_fn(self.dataset[index])

def get_return_transform(args):
    if args.dataset_name in ['clear10', 'clear100_cvpr2022']:
        return lambda x: x[:2]
    else:
        raise NotImplementedError

def get_trainer(args):
    if args.trainer == 'default':
        return default_trainer
    elif args.trainer == 'ensemble':
        return ensemble_trainer

def get_dataloader(args, train_stream, index):
    data_set = train_stream[index].dataset.train()
    train_loader = torch.utils.data.DataLoader(data_set, 
                            batch_size=args.batch_size, shuffle=True)
    return train_loader

def get_al_size(args, data_size:int=None):
    if args.query_size <= 1.:
        assert data_size is not None
        size = int(data_size * args.query_size)
    else:
        size = args.query_size
    return max(data_size, size)
    

def train(args, loader, model, criterion):
    optimizer, scheduler = get_optimizer_schedular(model, args.optimizer_config, args.scheduler_config)
    for epoch in range(args.num_epochs):
        acc_ = 0
        for _, data in enumerate(loader):
            input, target, _ = data
            optimizer.zero_grad()
            input = input.cuda()
            target = target.cuda()
            pred = model(input)
            loss = criterion(pred, target)
            loss.backward()
            # print(loss)
            acc_ += (torch.sum(torch.eq(torch.max(pred, 1)[1], 
                            target)) / len(pred)).item()
            optimizer.step()
        acc_ = acc_/len(loader)
        print(f'training accuracy for epoch {epoch} is {acc_}')
        scheduler.step()
    return model


def main(args):
    
    # For CLEAR dataset setup
    print(
        f"This script will train on {args.dataset_name}. "
        "You may change the dataset in command argument or config file."
    )

    # please refer to paper for discussion on streaming v.s. iid protocol
    EVALUATION_PROTOCOL = args.evaluation_protocol
    # EVALUATION_PROTOCOL = "streaming"  # trainset = testset per timestamp
    # EVALUATION_PROTOCOL = "iid"  # 7:3 trainset_size:testset_size


    # Paths for saving datasets/models/results/log files
    print(
        f"The dataset will be saved at {Path(args.dataset_path).resolve()}. "
        f"The model will be saved at {Path(args.save_path).resolve()}. "
        f"You may change this path in config.py."
    )
    ROOT = Path(args.dataset_path)
    DATA_ROOT = ROOT / args.dataset_name
    SAVE_ROOT = Path(args.save_path)
    MODEL_ROOT = SAVE_ROOT / "models"
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    model = get_model(args)

    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.RandomCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ]
    )


    # log to Tensorboard
    tb_logger = TensorboardLogger(SAVE_ROOT)

    # log to text file
    text_logger = TextLogger(open(SAVE_ROOT / "log.txt", "w+"))

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(
            num_classes=args.num_classes, save_image=False, stream=True
        ),
        disk_usage_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loggers=[interactive_logger, text_logger, tb_logger],
    )

    if EVALUATION_PROTOCOL == "streaming":
        seed = None
    else:
        seed = args.seed

    scenario = CLEAR(
        data_name=args.dataset_name,
        evaluation_protocol=EVALUATION_PROTOCOL,
        feature_type=None,
        seed=seed,
        train_transform=train_transform,
        eval_transform=test_transform,
        dataset_root=DATA_ROOT,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_fn = get_trainer(args)

    optimizer, scheduler = get_optimizer_schedular(model, args.optimizer_config, args.scheduler_config)
    criterion = torch.nn.CrossEntropyLoss()
    plugin_list = [LRSchedulerPlugin(scheduler)]
    cl_strategy = Naive(
        model,
        optimizer,
        criterion,
        train_mb_size=args.batch_size,
        train_epochs=args.num_epochs,
        eval_mb_size=args.batch_size,
        evaluator=eval_plugin,
        device=device,
        plugins=plugin_list,
    )
    
    active_query_cls = NAME_TO_CLS[args.query_type]
    
    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    print("Current protocol : ", EVALUATION_PROTOCOL)
    for index, experience in enumerate(scenario.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        if args.active_learning:
            # Select Query and Get Labels
            train_dataset = get_dataset(scenario.train_stream, index)
            query_dataset = QDataset(train_dataset, get_return_transform(args))
            query_size = get_al_size(args, len(query_dataset))
            
            pool = ActivePool(train_set=train_dataset, query_set=query_dataset, test_set=None, batch_size=args.batch_size)
            sampler = active_query_cls(model=model, pool=pool, size=query_size, device=device) \
                if index > 0 else RandomSampling(model=model, pool=pool, size=query_size, device=device)
            queries = sampler()
            pool.update(queries)
            
            train_loader = pool.get_labeled_dataloader()
        else:
            train_loader = get_dataloader(args, scenario.train_stream, index)
        
        exp_results = train_fn(
            train, args, train_loader, scenario.test_stream, model, 
            criterion=criterion, cl_strategy=cl_strategy, save_path=MODEL_ROOT, episode_idx=index)
        
        results.append(exp_results)

    # generate accuracy matrix
    num_timestamp = len(results)
    accuracy_matrix = np.zeros((num_timestamp, num_timestamp))
    for train_idx in range(num_timestamp):
        for test_idx in range(num_timestamp):
            accuracy_matrix[train_idx][test_idx] = results[train_idx][
                f"Top1_Acc_Stream/eval_phase/test_stream/Task{test_idx:03d}"]
    print('Accuracy_matrix : ')
    print(accuracy_matrix)
    metric = CLEARMetric().get_metrics(accuracy_matrix)
    print(metric)

    metric_log = open(SAVE_ROOT / "metric_log.txt", "w+")
    metric_log.write(
        f"Protocol: {EVALUATION_PROTOCOL} "
        f"Seed: {seed} "
    )
    json.dump(accuracy_matrix.tolist(), metric_log, indent=6)
    json.dump(metric, metric_log, indent=6)
    metric_log.close()


def create_and_parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser("Continual active learning")

    parser.add_argument('-c', '--config', type=str, required=False)

    parser.add_argument('--run_name',     type=str, default='ensemble')
    parser.add_argument('--save_path',    type=str, default='saved/')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tiny'])
    parser.add_argument('--dataset_path', type=str, default='datasets')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arch', type=str, default="resnet18", choices=["resnet18", "resnet50", "vgg16", "densenet121"])
    
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--resume_from', type=str, required=None, help='Resume CAL from the saved path.')

    parser = add_training_args(parser)
    parser = add_query_args(parser)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = create_and_parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            args_dict = json.load(f)
        args.__dict__.update(args_dict)

    print(vars(args))

    args.run_name = f"{args.dataset_name}_{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.save_path = os.path.join(args.save_path, args.run_name)
    os.makedirs(args.save_path, exist_ok=True)
    print(f"Experiment results will be saved to {args.save_path}")

    write_json(vars(args), os.path.join(args.save_path, "config.json"))

    main(args)