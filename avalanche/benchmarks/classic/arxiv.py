from avalanche.benchmarks.datasets.arxiv import ARXIVDataset
from avalanche.benchmarks.scenarios.generic_benchmark_creation import (
    create_generic_benchmark_from_paths,
    create_generic_benchmark_from_tensor_lists,
)
from typing import Optional, Any, Union, List
from pathlib import Path

def ARXIV(
    data_name: str = "arxiv",
    evaluation_protocol: str = "streaming",
    seed: Optional[int] = None,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
    dataset_root: Optional[Union[str, Path]] = None,
    bucket_list: Optional[List[str]] = None,
):
    arxiv_dataset_train = ARXIVDataset(
        root=dataset_root,
        data_name=data_name,
        split="train",
        seed=seed,
        bucket_list=bucket_list
    )
    arxiv_dataset_test = ARXIVDataset(
        root=dataset_root,
        data_name=data_name,
        split="test",
        seed=seed,
        bucket_list=bucket_list
    )
    train_samples = arxiv_dataset_train
    test_samples = arxiv_dataset_test
    
    benchmark_obj = create_generic_benchmark_from_tensor_lists(
        train_samples,
        test_samples,
        task_labels=list(range(len(train_samples))),
        complete_test_set_only=False,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    return benchmark_obj

if __name__ == "__main__":
    data_name = "arxiv"
    root = "/workspace/DB/wild-time-data"
    benchmark_instance = ARXIV(
        dataset_root=root,
        bucket_list=[2007]
    )
    benchmark_instance.train_stream[0]