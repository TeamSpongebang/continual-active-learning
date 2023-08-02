import copy
import torch

def default_trainer(
    train_fn, args, dataloader, test_stream, model, criterion,
    cl_strategy, save_path, episode_idx):
    
    model = train_fn(args, dataloader, model, criterion=criterion)
    
    cl_strategy.model = copy.deepcopy(model)
    torch.save(
            model.state_dict(),
            str(save_path / f"model{str(episode_idx).zfill(2)}.pth")
        )
    
    print("Training completed")
    print(
        "Computing accuracy on the whole test set with evaluation protocol"
    )
    exp_results = {}
    for tid, texp in enumerate(test_stream):
        exp_results.update(cl_strategy.eval(texp))
        
    return model, exp_results