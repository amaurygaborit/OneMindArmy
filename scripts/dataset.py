import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset

class OneMindArmyDataset(Dataset):
    """
    Highly optimized PyTorch Dataset for loading OneMindArmy binary streams.
    Implements 'Lazy Initialization' to prevent RAM explosions with num_workers > 0 on Windows/Linux.
    Safely ignores incomplete trailing bytes caused by abrupt C++ termination.
    """
    def __init__(self, bin_path: str):
        super().__init__()
        self.bin_path = bin_path
        
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"[Dataset] Binary file not found: {bin_path}")
            
        meta_path = f"{bin_path}.meta.json"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"[Dataset] Metadata not found: {meta_path}. Run C++ MetaExport first!")

        # 1. Load exact C++ compiled constants
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)

        self.action_space = self.meta["actionSpace"]
        self.num_players = self.meta["numPlayers"]
        self.nn_input_size = self.meta["nnInputSize"]
        self.cpp_struct_size = self.meta.get("sizeofTrainingSample", None)

        # 2. Define the strict Binary Layout (Matching C++ struct exactly)
        data_fields = [
            ('nn_input', np.float32, (self.nn_input_size,)),
            ('policy', np.float32, (self.action_space,)),
            ('legal_mask', np.float32, (self.action_space,)), # NOUVEAU: Le masque des coups légaux
            ('result', np.float32, (self.num_players,))
        ]
        
        # Calcul de la taille théorique (en octets, float32 = 4 octets)
        data_size_bytes = (self.nn_input_size + (self.action_space * 2) + self.num_players) * 4

        # 3. Absorb C++ Padding
        if self.cpp_struct_size and self.cpp_struct_size > data_size_bytes:
            padding_bytes = self.cpp_struct_size - data_size_bytes
            data_fields.append(('padding', np.uint8, (padding_bytes,)))
            
        self.sample_dtype = np.dtype(data_fields)

        # 4. LAZY LOADING & CORRUPTION PROTECTION
        # We explicitly DO NOT open the file here to prevent memory leaks in multiprocessing.
        self.data = None
        
        # We calculate the exact number of valid samples
        file_size = os.path.getsize(bin_path)
        excess_bytes = file_size % self.sample_dtype.itemsize
        
        if excess_bytes != 0:
            print(f"[Dataset] Notice: Ignoring {excess_bytes} trailing bytes (incomplete sample from abrupt C++ stop).")
            
        self.num_samples = file_size // self.sample_dtype.itemsize

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 5. OPEN ON FIRST READ (Worker Context)
        # This ensures each PyTorch worker opens its own lightweight file handle.
        if self.data is None:
            # En forçant la 'shape', Numpy ignorera les octets corrompus à la fin du fichier.
            self.data = np.memmap(
                self.bin_path, 
                dtype=self.sample_dtype, 
                mode='r', 
                shape=(self.num_samples,)
            )
            
        sample = self.data[idx]
        
        # np.copy is required to transfer data from the read-only mmap to writable PyTorch tensors
        state_tensor  = torch.from_numpy(np.copy(sample['nn_input']))
        policy_tensor = torch.from_numpy(np.copy(sample['policy']))
        mask_tensor   = torch.from_numpy(np.copy(sample['legal_mask'])) # Extraction du masque
        result_tensor = torch.from_numpy(np.copy(sample['result']))

        return state_tensor, policy_tensor, mask_tensor, result_tensor