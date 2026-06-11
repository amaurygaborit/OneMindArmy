import os
import json
import glob
import torch
import numpy as np
from torch.utils.data import Dataset

class OneMindArmyDataset(Dataset):
    """
    Zero-Copy Binary Dataset Loader.

    Design Intent:
    Directly maps Python numpy types to the underlying C++ #pragma pack(1) memory 
    layout. Uses `np.memmap` to lazy-load massive datasets (GBs/TBs) straight from 
    disk into VRAM without exhausting system RAM.
    """

    def __init__(self, bin_path: str):
        super().__init__()
        self.bin_path = bin_path

        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"[Dataset] Binary file not found: {bin_path}")

        # ------------------------------------------------------------------
        # 1. Metadata Resolution
        # Identifies the C++ layout dimensions required to unpack the binary.
        # ------------------------------------------------------------------
        dir_name   = os.path.dirname(os.path.abspath(bin_path))
        meta_files = sorted(glob.glob(os.path.join(dir_name, "*.meta.json")))

        if not meta_files:
            raise FileNotFoundError(f"[Dataset] No *.meta.json found in {dir_name}.")

        if len(meta_files) == 1:
            meta_path = meta_files[0]
        else:
            game_name = os.path.basename(dir_name)
            canonical = os.path.join(dir_name, f"{game_name}_training_data.bin.meta.json")
            meta_path = canonical if canonical in meta_files else meta_files[0]

        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.action_space    = self.meta["actionSpace"]
        self.num_players     = self.meta["numPlayers"]
        self.nn_input_size   = self.meta["nnInputSize"]
        self.cpp_struct_size = self.meta.get("sizeofTrainingSample", None)

        # Calculates minimum byte size required to unpack the bit-packed boolean mask.
        self.mask_bytes_size = (self.action_space + 7) // 8

        # ------------------------------------------------------------------
        # 2. Strict C++ Memory Mapping
        # WDL Target requires 3 floats per player (Win, Draw, Loss) to handle 
        # N-player zero-sum and non-zero-sum game outcomes.
        # ------------------------------------------------------------------
        wdl_size = self.num_players * 3  

        data_fields = [
            ("nn_input",   np.float32, (self.nn_input_size,)),
            ("policy",     np.float32, (self.action_space,)),
            ("wdl_target", np.float32, (wdl_size,)),          
            ("legal_mask", np.uint8,   (self.mask_bytes_size,)),
        ]

        data_size_bytes = (
            (self.nn_input_size + self.action_space + wdl_size) * 4
            + self.mask_bytes_size
        )

        # Absorbs OS/Compiler specific struct padding from C++ to prevent 
        # progressive offset corruption as the file is iterated.
        if self.cpp_struct_size and self.cpp_struct_size > data_size_bytes:
            padding_bytes = self.cpp_struct_size - data_size_bytes
            data_fields.append(("_padding", np.uint8, (padding_bytes,)))

        self.sample_dtype = np.dtype(data_fields)

        # ------------------------------------------------------------------
        # 3. File Truncation Resilience
        # Protects against crash-induced incomplete trailing bytes.
        # ------------------------------------------------------------------
        self.data = None
        file_size    = os.path.getsize(bin_path)
        excess_bytes = file_size % self.sample_dtype.itemsize

        if excess_bytes != 0:
            print(f"[Dataset] {os.path.basename(bin_path)}: ignoring "
                  f"{excess_bytes} trailing byte(s) (incomplete sample).")

        self.num_samples = file_size // self.sample_dtype.itemsize

        if self.num_samples == 0:
            print(f"[Dataset] Warning: {os.path.basename(bin_path)} has 0 valid samples!")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Defers memmap initialization to the worker process layer to prevent 
        # multiprocessing serialization locks in PyTorch DataLoaders.
        if self.data is None:
            self.data = np.memmap(
                self.bin_path, dtype=self.sample_dtype,
                mode="r", shape=(self.num_samples,))

        sample = self.data[idx]

        state_tensor  = torch.from_numpy(np.array(sample["nn_input"],   dtype=np.float32))
        policy_tensor = torch.from_numpy(np.array(sample["policy"],     dtype=np.float32))
        wdl_tensor    = torch.from_numpy(np.array(sample["wdl_target"], dtype=np.float32))

        # Re-inflates the C++ byte mask back into a float32 tensor mask.
        # bitorder='little' mirrors the C++ `1 << (i % 8)` packing logic.
        packed_mask   = np.array(sample["legal_mask"], dtype=np.uint8)
        unpacked_mask = np.unpackbits(packed_mask, bitorder="little")[: self.action_space]
        mask_tensor   = torch.from_numpy(unpacked_mask.copy()).float()

        return state_tensor, policy_tensor, mask_tensor, wdl_tensor