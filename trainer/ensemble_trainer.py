import os
import copy
from typing import List
import torch

def ensemble_trainer(
    train_fn, args, dataloader, test_stream, model, criterion,
    cl_strategy, save_path:str, episode_idx:int=0):
    
    ckpt_paths = []
    
    for ens in range(args.num_ensembles):
        if episode_idx == 0:
            if args.ensemble_config["finetune"]:
                # Use same model but shuffle dataloader
                if ens == 0:
                    org_model = copy.deepcopy(model)
                else:
                    model = copy.deepcopy(org_model)
            else:
                # Random initialization for ensemble members
                for module in model.modules():
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()
        else:
            # Load from last episode
            ckpt_file = str(save_path / f"member_{ens}_{str(episode_idx-1).zfill(2)}.pth")
            ckpt = torch.load(ckpt_file)
            model.load_state_dict(ckpt["state_dict"])
        
        # Train
        model = train_fn(args, dataloader, model, criterion=criterion)
        
        # Save members
        ckpt_file = str(save_path / f"member_{ens}_{str(episode_idx).zfill(2)}.pth")
        torch.save({"state_dict":model.state_dict()}, ckpt_file)
        ckpt_paths.append(ckpt_file)
    
    cl_strategy.model = EnsembleEvaluator(model, ckpt_paths=ckpt_paths)
    
    print("Training completed")
    print(
        "Computing accuracy on the whole test set with evaluation protocol"
    )
    exp_results = {}
    # for tid, texp in enumerate(test_stream):
    #     exp_results.update(cl_strategy.eval(texp))
    
    return (model, ckpt_paths), exp_results # model is unused.
    

class EnsembleEvaluator(torch.nn.Module):
    def __init__(self, model, ckpt_paths:List[str]):
        super().__init__()
        self.members = [copy.deepcopy(model) for _ in ckpt_paths]
        for member, ckpt_path in zip(self.members, ckpt_paths):
            member.load_state_dict(torch.load(ckpt_path)["state_dict"])
        
    def eval(self):
        for member in self.members:
            member.eval()

    def forward(self, inputs):
        logits = []
        with torch.no_grad():
            for member in self.members:
                logit = member(inputs)
                logits.append(logit)
            logits = torch.stack(logits)
            probs = torch.softmax(logits, dim=-1)
            mean_probs = probs.mean(axis=0)
            mean_logits = torch.log(mean_probs / (1-mean_probs))
        return mean_logits
