import torch
import torchvision
import torch.nn as nn

from trainer.ensemble_trainer import ensemble_trainer
from trainer.trainer import default_trainer


def make_scheduler(optimizer, scheduler_args):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_args)
    return scheduler

def get_model(args):
    if args.dataset_name == 'clear10':
        num_classes = 11
    elif args.dataset_name == 'clear100':
        num_classes = 100
    else:
        NotImplementedError

    if args.arch == 'resnet18':
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1" if args.use_pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif args.arch == 'resnet50':
        model = torchvision.models.resnet50(weights="IMAGENET1K_V1" if args.use_pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif args.arch == 'vit16':
        model = torchvision.models.vit_b_16(weights="IMAGENET1K_V1" if args.use_pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    elif args.arch == 'vit32':
        model = torchvision.models.vit_b_32(weights="IMAGENET1K_V1" if args.use_pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    else:
        raise NotImplementedError(f"Architecture {args.arch} not supported.")

    return model

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
    if args.dataset_name in ['clear10', 'clear100']:
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
    return min(data_size, size)