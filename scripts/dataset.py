import os
import json
import glob
import torch
import numpy as np
from torch.utils.data import Dataset


class OneMindArmyDataset(Dataset):
    """
    Highly optimized PyTorch Dataset for loading OneMindArmy binary streams.
    Includes support for the Game-Agnostic Multi-Player WDL Target structure.
    """

    def __init__(self, bin_path: str):
        super().__init__()
        self.bin_path = bin_path

        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"[Dataset] Binary file not found: {bin_path}")

        # 1. Locate the meta.json
        dir_name   = os.path.dirname(os.path.abspath(bin_path))
        meta_files = sorted(glob.glob(os.path.join(dir_name, "*.meta.json")))

        if not meta_files:
            raise FileNotFoundError(
                f"[Dataset] No *.meta.json file found in {dir_name}. "
                "Run C++ MetaExport first!"
            )

        if len(meta_files) == 1:
            meta_path = meta_files[0]
        else:
            game_name   = os.path.basename(dir_name)
            canonical   = os.path.join(dir_name, f"{game_name}_training_data.bin.meta.json")
            if canonical in meta_files:
                meta_path = canonical
            else:
                meta_path = meta_files[0]
                print(
                    f"[Dataset] Warning: multiple meta.json files found in {dir_name}. "
                    f"Using '{os.path.basename(meta_path)}'."
                )

        # 2. Load dimensions from meta.json
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.action_space    = self.meta["actionSpace"]
        self.num_players     = self.meta["numPlayers"]
        self.nn_input_size   = self.meta["nnInputSize"]
        self.cpp_struct_size = self.meta.get("sizeofTrainingSample", None)

        self.mask_bytes_size = (self.action_space + 7) // 8

        # 3. Build the numpy dtype that mirrors the C++ struct layout exactly
        # NEW WDL LAYOUT: num_players * 3 floats instead of num_players * 1
        self.wdl_size = self.num_players * 3

        data_fields = [
            ("nn_input",   np.float32, (self.nn_input_size,)),
            ("policy",     np.float32, (self.action_space,)),
            ("wdl_target", np.float32, (self.wdl_size,)), # <--- CHANGED HERE
            ("legal_mask", np.uint8,   (self.mask_bytes_size,)),
        ]

        data_size_bytes = (
            (self.nn_input_size + self.action_space + self.wdl_size) * 4
            + self.mask_bytes_size
        )

        if self.cpp_struct_size and self.cpp_struct_size > data_size_bytes:
            padding_bytes = self.cpp_struct_size - data_size_bytes
            data_fields.append(("_padding", np.uint8, (padding_bytes,)))

        self.sample_dtype = np.dtype(data_fields)

        # 4. Determine sample count
        self.data = None
        file_size    = os.path.getsize(bin_path)
        excess_bytes = file_size % self.sample_dtype.itemsize

        if excess_bytes != 0:
            print(
                f"[Dataset] {os.path.basename(bin_path)}: ignoring {excess_bytes} trailing "
                "byte(s) (incomplete sample from abrupt C++ stop)."
            )

        self.num_samples = file_size // self.sample_dtype.itemsize

        if self.num_samples == 0:
            print(f"[Dataset] Warning: {os.path.basename(bin_path)} contains 0 valid samples!")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.data is None:
            self.data = np.memmap(
                self.bin_path,
                dtype=self.sample_dtype,
                mode="r",
                shape=(self.num_samples,),
            )

        sample = self.data[idx]

        state_tensor  = torch.from_numpy(np.array(sample["nn_input"],   dtype=np.float32))
        policy_tensor = torch.from_numpy(np.array(sample["policy"],     dtype=np.float32))
        
        # Reshape the WDL target to [num_players, 3] for easier loss calculation
        wdl_flat   = np.array(sample["wdl_target"], dtype=np.float32)
        wdl_tensor = torch.from_numpy(wdl_flat.reshape(self.num_players, 3)) 

        packed_mask   = np.array(sample["legal_mask"], dtype=np.uint8)
        unpacked_mask = np.unpackbits(packed_mask, bitorder="little")[: self.action_space]
        mask_tensor   = torch.from_numpy(unpacked_mask.copy()).float()

        return state_tensor, policy_tensor, mask_tensor, wdl_tensor