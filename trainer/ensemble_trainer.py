import os
import copy
from typing import List
import torch

def ensemble_trainer(
    train_fn, args, dataloader, test_stream, model, criterion, 
    cl_strategy, save_path:str, episode_idx:int=0, num_ensemble:int=5):
    
    ckpt_paths = []
    
    for ens in range(num_ensemble):
        if episode_idx == 0:
            # Random initialization for ensemble members
            member = copy.deepcopy(model)
            for module in member.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
        else:
            # Load from last episode
            ckpt_file = str(save_path / f"member_{ens}_{str(episode_idx-1).zfill(2)}.pth")
            ckpt = torch.load(ckpt_file)
            member = copy.deepcopy(model)
            member.load_state_dict(ckpt)
        
        # Train
        member = train_fn(args, dataloader, member, criterion=criterion)
        
        # Save members
        ckpt_file = str(save_path / f"member_{ens}_{str(episode_idx).zfill(2)}.pth")
        torch.save(member.state_dict(), ckpt_file)
        ckpt_paths.append(ckpt_file)
    
    cl_strategy.model = member # EnsembleEvaluator(model, ckpt_paths=ckpt_paths)
    
    print("Training completed")
    print(
        "Computing accuracy on the whole test set with evaluation protocol"
    )
    exp_results = {}
    for tid, texp in enumerate(test_stream):
        exp_results.update(cl_strategy.eval(texp))
    
    return exp_results
    

class EnsembleEvaluator(torch.nn.Module):
    def __init__(self, model, ckpt_paths:List[str]):
        super().__init__()
        self.members = [copy.deepcopy(model) for _ in ckpt_paths]
        for member, ckpt_path in zip(self.members, ckpt_paths):
            member.load_state_dict(torch.load(ckpt_path))
        
    def eval(self):
        for member in self.members:
            member.eval()

    def forward(self, inputs):
        logits = []
        with torch.no_grad():
            import pdb;pdb.set_trace()
            for member in self.members:
                logit = member(inputs)
                logits.append(logit)
            import pdb;pdb.set_trace()
            logits = torch.stack(logits)
            probs = torch.softmax(logits, dim=-1)
            mean_probs = probs.mean(axis=0)
            mean_logits = torch.log(mean_probs / (1-mean_probs))
        return mean_logits
