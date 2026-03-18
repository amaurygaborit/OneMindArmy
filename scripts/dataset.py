import os
import json
import glob
import torch
import numpy as np
from torch.utils.data import Dataset


class OneMindArmyDataset(Dataset):
    """
    Highly optimized PyTorch Dataset for loading OneMindArmy binary streams.

    Design principles:
    - Lazy initialization: the memmap is opened only on the first __getitem__ call,
      inside the worker process. This prevents RAM explosions when num_workers > 0.
    - Corruption protection: trailing incomplete bytes are silently ignored.
    - Bit-packed legal mask: unpacked on-the-fly for memory efficiency.
    - Game-agnostic: works for any number of players and any action space.
    """

    def __init__(self, bin_path: str):
        super().__init__()
        self.bin_path = bin_path

        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"[Dataset] Binary file not found: {bin_path}")

        # ------------------------------------------------------------------
        # 1. Locate the meta.json that lives in the same directory.
        #    Strategy: prefer the file whose name matches the binary's directory
        #    name (i.e. the canonical "<game>_training_data.bin.meta.json").
        #    If none matches, fall back to the only .meta.json present.
        #    Raise clearly if ambiguous.
        # ------------------------------------------------------------------
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
            # More than one meta file: try to find the canonical one
            # (named after the game directory)
            game_name   = os.path.basename(dir_name)
            canonical   = os.path.join(dir_name, f"{game_name}_training_data.bin.meta.json")
            if canonical in meta_files:
                meta_path = canonical
            else:
                # Give up and use the first one, but warn loudly
                meta_path = meta_files[0]
                print(
                    f"[Dataset] Warning: multiple meta.json files found in {dir_name}. "
                    f"Using '{os.path.basename(meta_path)}'. "
                    "Consider keeping only one meta file per game directory."
                )

        # ------------------------------------------------------------------
        # 2. Load dimensions from meta.json
        # ------------------------------------------------------------------
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.action_space    = self.meta["actionSpace"]
        self.num_players     = self.meta["numPlayers"]
        self.nn_input_size   = self.meta["nnInputSize"]
        self.cpp_struct_size = self.meta.get("sizeofTrainingSample", None)

        # Number of bytes needed to store the bit-packed legal mask
        self.mask_bytes_size = (self.action_space + 7) // 8

        # ------------------------------------------------------------------
        # 3. Build the numpy dtype that mirrors the C++ struct layout exactly
        # ------------------------------------------------------------------
        data_fields = [
            ("nn_input",   np.float32, (self.nn_input_size,)),
            ("policy",     np.float32, (self.action_space,)),
            ("result",     np.float32, (self.num_players,)),
            ("legal_mask", np.uint8,   (self.mask_bytes_size,)),
        ]

        # Theoretical byte count (float32 = 4 bytes, uint8 = 1 byte)
        data_size_bytes = (
            (self.nn_input_size + self.action_space + self.num_players) * 4
            + self.mask_bytes_size
        )

        # Absorb any C++ struct padding so memmap alignment stays correct
        if self.cpp_struct_size and self.cpp_struct_size > data_size_bytes:
            padding_bytes = self.cpp_struct_size - data_size_bytes
            data_fields.append(("_padding", np.uint8, (padding_bytes,)))

        self.sample_dtype = np.dtype(data_fields)

        # ------------------------------------------------------------------
        # 4. Determine sample count — silently drop any trailing incomplete bytes
        # ------------------------------------------------------------------
        # Data is not loaded here — memmap is created lazily in __getitem__
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

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Open the memmap on the very first access inside this worker process.
        # This is the standard pattern for lazy loading with num_workers > 0.
        if self.data is None:
            self.data = np.memmap(
                self.bin_path,
                dtype=self.sample_dtype,
                mode="r",
                shape=(self.num_samples,),
            )

        sample = self.data[idx]

        # Plain float tensors
        state_tensor  = torch.from_numpy(np.array(sample["nn_input"],  dtype=np.float32))
        policy_tensor = torch.from_numpy(np.array(sample["policy"],    dtype=np.float32))
        result_tensor = torch.from_numpy(np.array(sample["result"],    dtype=np.float32))

        # Bit-packed legal mask → float32 tensor of 0s and 1s
        # bitorder='little' matches C++'s `(1 << (i % 8))` packing convention
        packed_mask   = np.array(sample["legal_mask"], dtype=np.uint8)
        unpacked_mask = np.unpackbits(packed_mask, bitorder="little")[: self.action_space]
        mask_tensor   = torch.from_numpy(unpacked_mask.copy()).float()

        return state_tensor, policy_tensor, mask_tensor, result_tensor
