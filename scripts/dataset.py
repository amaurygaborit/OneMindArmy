import os
import json
import torch
import numpy as np
import glob
from torch.utils.data import Dataset

class OneMindArmyDataset(Dataset):
    """
    Highly optimized PyTorch Dataset for loading OneMindArmy binary streams.
    Implements 'Lazy Initialization' to prevent RAM explosions with num_workers > 0 on Windows/Linux.
    Safely ignores incomplete trailing bytes caused by abrupt C++ termination.
    Unpacks bit-packed legal move masks on the fly for extreme memory efficiency.
    """
    def __init__(self, bin_path: str):
        super().__init__()
        self.bin_path = bin_path
        
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"[Dataset] Binary file not found: {bin_path}")
            
        # 1. Résolution dynamique du meta.json (pour supporter iteration_XXXX.bin)
        dir_name = os.path.dirname(bin_path)
        meta_files = glob.glob(os.path.join(dir_name, "*.meta.json"))
        
        if not meta_files:
            raise FileNotFoundError(f"[Dataset] Metadata not found in {dir_name}. Run C++ MetaExport first!")
        meta_path = meta_files[0] # On prend le premier meta.json trouvé dans le dossier

        # 2. Load exact C++ compiled constants
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)

        self.action_space = self.meta["actionSpace"]
        self.num_players = self.meta["numPlayers"]
        self.nn_input_size = self.meta["nnInputSize"]
        self.cpp_struct_size = self.meta.get("sizeofTrainingSample", None)

        # Calcul du nombre d'octets du bitset (arrondi au supérieur)
        self.mask_bytes_size = (self.action_space + 7) // 8

        # 3. Define the strict Binary Layout (Match EXACT de l'ordre du struct C++)
        data_fields = [
            ('nn_input', np.float32, (self.nn_input_size,)),
            ('policy', np.float32, (self.action_space,)),
            ('result', np.float32, (self.num_players,)),
            ('legal_mask', np.uint8, (self.mask_bytes_size,)) # Le masque ultra-compressé à la fin
        ]
        
        # Calcul de la taille théorique (float32 = 4 octets, uint8 = 1 octet)
        data_size_bytes = ((self.nn_input_size + self.action_space + self.num_players) * 4) + (self.mask_bytes_size * 1)

        # 4. Absorb C++ Padding
        if self.cpp_struct_size and self.cpp_struct_size > data_size_bytes:
            padding_bytes = self.cpp_struct_size - data_size_bytes
            data_fields.append(('padding', np.uint8, (padding_bytes,)))
            
        self.sample_dtype = np.dtype(data_fields)

        # 5. LAZY LOADING & CORRUPTION PROTECTION
        self.data = None
        
        file_size = os.path.getsize(bin_path)
        excess_bytes = file_size % self.sample_dtype.itemsize
        
        if excess_bytes != 0:
            print(f"[Dataset] Notice: Ignoring {excess_bytes} trailing bytes (incomplete sample from abrupt C++ stop).")
            
        self.num_samples = file_size // self.sample_dtype.itemsize

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # OPEN ON FIRST READ (Worker Context)
        if self.data is None:
            self.data = np.memmap(
                self.bin_path, 
                dtype=self.sample_dtype, 
                mode='r', 
                shape=(self.num_samples,)
            )
            
        sample = self.data[idx]
        
        # Extraction des tenseurs float classiques
        state_tensor  = torch.from_numpy(np.copy(sample['nn_input']))
        policy_tensor = torch.from_numpy(np.copy(sample['policy']))
        result_tensor = torch.from_numpy(np.copy(sample['result']))

        # --- DÉCOMPRESSION DU MASQUE BITS ---
        # On lit les octets
        packed_mask = sample['legal_mask']
        # On déploie les bits (little-endian pour matcher le (1 << i%8) du C++)
        unpacked_mask = np.unpackbits(packed_mask, bitorder='little')
        # On coupe les potentiels bits vides à la toute fin
        unpacked_mask = unpacked_mask[:self.action_space]
        # On convertit le tableau de 0 et de 1 en FloatTensor pour PyTorch
        mask_tensor = torch.from_numpy(unpacked_mask.copy()).float()

        return state_tensor, policy_tensor, mask_tensor, result_tensor