import os
import json
import glob
import torch
import numpy as np
from torch.utils.data import Dataset


class OneMindArmyDataset(Dataset):
    """
    PyTorch Dataset for OneMindArmy binary training files.

    Binary layout (C++ TrainingSample, #pragma pack(1)):
      nnInput      : float32 × nnInputSize
      policy       : float32 × actionSpace
      wdlTarget    : float32 × numPlayers * 3   ← WDL per player (W, D, L)
      legalMask    : uint8   × ceil(actionSpace / 8)
      [padding]    : uint8   × (sizeofTrainingSample - computed_size)  if any

    The WDL target has numPlayers * 3 floats, NOT numPlayers.
    This matches C++ TrainingSample::wdlTarget = std::array<float, kNumPlayers * 3>.
    """

    def __init__(self, bin_path: str):
        super().__init__()
        self.bin_path = bin_path

        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"[Dataset] Binary file not found: {bin_path}")

        # ------------------------------------------------------------------
        # 1. Locate meta.json
        # ------------------------------------------------------------------
        dir_name   = os.path.dirname(os.path.abspath(bin_path))
        meta_files = sorted(glob.glob(os.path.join(dir_name, "*.meta.json")))

        if not meta_files:
            raise FileNotFoundError(
                f"[Dataset] No *.meta.json found in {dir_name}.")

        if len(meta_files) == 1:
            meta_path = meta_files[0]
        else:
            game_name = os.path.basename(dir_name)
            canonical = os.path.join(dir_name,
                                     f"{game_name}_training_data.bin.meta.json")
            if canonical in meta_files:
                meta_path = canonical
            else:
                meta_path = meta_files[0]
                print(f"[Dataset] Warning: multiple meta.json in {dir_name}, "
                      f"using '{os.path.basename(meta_path)}'.")

        # ------------------------------------------------------------------
        # 2. Load dimensions
        # ------------------------------------------------------------------
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.action_space    = self.meta["actionSpace"]
        self.num_players     = self.meta["numPlayers"]
        self.nn_input_size   = self.meta["nnInputSize"]
        self.cpp_struct_size = self.meta.get("sizeofTrainingSample", None)

        # Bit-packed legal mask size
        self.mask_bytes_size = (self.action_space + 7) // 8

        # ------------------------------------------------------------------
        # 3. Build numpy dtype mirroring C++ TrainingSample layout exactly
        #
        # FIX: wdlTarget is numPlayers * 3 floats (Win, Draw, Loss per player)
        #      NOT numPlayers floats.
        #      Using numPlayers would shift all subsequent fields by
        #      (numPlayers * 2 * 4) bytes, making the legal mask garbage.
        # ------------------------------------------------------------------
        wdl_size = self.num_players * 3   # W, D, L per player

        data_fields = [
            ("nn_input",   np.float32, (self.nn_input_size,)),
            ("policy",     np.float32, (self.action_space,)),
            ("wdl_target", np.float32, (wdl_size,)),          # ← num_players * 3
            ("legal_mask", np.uint8,   (self.mask_bytes_size,)),
        ]

        # Theoretical byte count
        data_size_bytes = (
            (self.nn_input_size + self.action_space + wdl_size) * 4
            + self.mask_bytes_size
        )

        # Absorb C++ struct padding
        if self.cpp_struct_size and self.cpp_struct_size > data_size_bytes:
            padding_bytes = self.cpp_struct_size - data_size_bytes
            data_fields.append(("_padding", np.uint8, (padding_bytes,)))

        self.sample_dtype = np.dtype(data_fields)

        # ------------------------------------------------------------------
        # 4. Count samples (lazy loading — memmap opened in __getitem__)
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
        # Lazy memmap — opened once per worker process
        if self.data is None:
            self.data = np.memmap(
                self.bin_path, dtype=self.sample_dtype,
                mode="r", shape=(self.num_samples,))

        sample = self.data[idx]

        state_tensor  = torch.from_numpy(np.array(sample["nn_input"],   dtype=np.float32))
        policy_tensor = torch.from_numpy(np.array(sample["policy"],     dtype=np.float32))
        wdl_tensor    = torch.from_numpy(np.array(sample["wdl_target"], dtype=np.float32))

        # Bit-packed legal mask → float32 {0, 1}
        # bitorder='little' matches C++ (1 << (i % 8)) packing
        packed_mask   = np.array(sample["legal_mask"], dtype=np.uint8)
        unpacked_mask = np.unpackbits(packed_mask, bitorder="little")[: self.action_space]
        mask_tensor   = torch.from_numpy(unpacked_mask.copy()).float()

        return state_tensor, policy_tensor, mask_tensor, wdl_tensor