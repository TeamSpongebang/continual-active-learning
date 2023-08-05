from transformers import AutoTokenizer
from avalanche.benchmarks.utils import AvalancheDataset
from pathlib import Path
from typing import Optional, Union, List, Tuple
import h5py
import os
import torch

class ARXIVDataset(AvalancheDataset):
    def __init__(
            self,
            root: Optional[Union[str, Path]] = None,
            data_name: str = "arxiv",
            tokenizer: Optional[AutoTokenizer] = None,
            download: bool = False,
            verbose: bool = False,
            split: str = "train",
            seed: Optional[int] = None,
            bucket_list: Optional[List[int]] = None,
            max_length: int = 512,  # Maximum length of a sequence
    ):
        # Initialize parameters
        self.root = root
        self.data_name = data_name
        self.hdf5_file = self.data_name + '.hdf5'
        self.bucket_list = bucket_list
        self.split = split
        self.samples = []
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

        # Load data
        self._load_metadata()

    def _load_metadata(self):
        # Load data from hdf5 file 
        with h5py.File(os.path.join(self.root, self.hdf5_file), 'r') as f:
            for bucket_index in self.bucket_list:  # Assuming the dataset contains data for these years
                try:
                    year_group = f[str(bucket_index)][self.split]
                    texts = year_group['text'][:]
                    labels = year_group['labels'][:]
                    for text, label in zip(texts, labels):
                        self.samples.append((text, label))
                except Exception as e:
                    print(f"Error with year : {bucket_index}")
    
    def _encode_text(self, text):
        text = text.decode() 
        return self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        text, label = self.samples[index]
        # Encode the text using the provided tokenizer
        encoded_text = self._encode_text(text)
        return {
            'input_ids': encoded_text['input_ids'].flatten(),
            'attention_mask': encoded_text['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }




if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader
    import torch

    data_name = "arxiv"
    root = "/workspace/DB/wild-time-data"
    arxiv_trainset = ARXIVDataset(
        root=root,
        data_name=data_name,
        split="train",
        seed=None,
        bucket_list=[2007, 2008, 2009]
    )
    print(f"arxiv size (train): ", len(arxiv_trainset))
    dataloader = DataLoader(arxiv_trainset, batch_size=12)

    for batch in dataloader:
        print(batch)
        break