import torch
import collections
import os
import soundfile as sf
from torch.utils.data import DataLoader, Dataset
import numpy as np
from joblib import Parallel, delayed
import random
import csv

ASVFile = collections.namedtuple('ASVFile',
    ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])

class ASVDataset(Dataset):
    """
    Returns:
      x: waveform
      y: ground truth label => 0=spoof, 1=bonafide
      meta: an ASVFile tuple (speaker_id, file_name, path, sys_id, key)
      llm_label: nudge label in {0=don't know,1=spoof,2=bonafide}
        (with random noise injection during training, or CSV-based for eval)
    """

    def __init__(self, database_path=None, protocols_path=None, transform=None,
                 is_train=True, sample_size=None, is_logical=True,
                 feature_name=None, is_eval=False, eval_part=0,
                 llm_csv_path=None):
        track = 'LA'
        data_root = protocols_path
        assert feature_name is not None, 'must provide feature name'

        self.track = track
        self.is_logical = is_logical
        self.prefix = 'ASVspoof2019_{}'.format(track)

        # We unify possible sys IDs for LA
        self.sysid_dict = {
            '-': 0, 'A07': 1, 'A08': 2, 'A09': 3, 'A10': 4, 'A11': 5,
            'A12': 6, 'A13': 7, 'A14': 8, 'A15': 9, 'A16': 10, 'A17': 11,
            'A18': 12, 'A19': 13, 'A01': 14, 'A02': 15, 'A03': 16,
            'A04': 17, 'A05': 18, 'A06': 19
        }
        self.sysid_dict_inv = {v: k for k, v in self.sysid_dict.items()}

        self.data_root_dir = database_path
        self.data_root = data_root
        self.is_eval = is_eval

        # Decide whether train/dev/eval
        self.dset_name = 'eval' if is_eval else ('train' if is_train else 'dev')
        if is_eval:
            self.protocols_fname = 'eval.trl'
        elif is_train:
            self.protocols_fname = 'train.trn'
        else:
            self.protocols_fname = 'dev.trl'

        self.protocols_dir = os.path.join(self.data_root)
        self.files_dir = os.path.join(
            self.data_root_dir, f"{self.prefix}_{self.dset_name}", 'flac'
        )
        self.protocols_fname = os.path.join(
            self.protocols_dir,
            f'ASVspoof2019.{track}.cm.{self.protocols_fname}.txt'
        )

        self.cache_fname = f'cache_{self.dset_name}_{track}_{feature_name}.npy'
        self.transform = transform
        self.llm_csv_path = llm_csv_path

        # Parse official protocol
        if os.path.exists(self.cache_fname):
            self.data_x, self.data_y, self.data_sysid, self.files_meta = \
                torch.load(self.cache_fname)
            print('Dataset loaded from cache ', self.cache_fname)
        else:
            self.files_meta = self.parse_protocols_file(self.protocols_fname)
            data = [self.read_file(m) for m in self.files_meta]
            self.data_x, self.data_y, self.data_sysid = map(list, zip(*data))
            if self.transform:
                self.data_x = Parallel(n_jobs=4, prefer='threads')(
                    delayed(self.transform)(x) for x in self.data_x
                )
            torch.save((self.data_x, self.data_y, self.data_sysid, self.files_meta),
                       self.cache_fname)

        # Optionally subsample
        if sample_size:
            select_idx = np.random.choice(len(self.files_meta), size=(sample_size,),
                                          replace=True).astype(np.int32)
            self.files_meta = [self.files_meta[i] for i in select_idx]
            self.data_x = [self.data_x[i] for i in select_idx]
            self.data_y = [self.data_y[i] for i in select_idx]
            self.data_sysid = [self.data_sysid[i] for i in select_idx]

        # Optionally parse CSV for eval
        self.llm_map = {}
        if (self.llm_csv_path is not None) and self.is_eval:
            self.llm_map = self.parse_llm_csv(self.llm_csv_path)

        # Build final "nudge label"
        self.data_llm = []
        for idx, meta in enumerate(self.files_meta):
            # meta.key => 1 => bonafide, 0 => spoof
            if is_train:
                # 30% chance => 0 (dont know), else => 1(spoof) or 2(bonafide)
                base = 2 if meta.key == 1 else 1
                if random.random() < 0.3:
                    label_leak = 0
                else:
                    label_leak = base
            elif not is_eval:
                # dev => ground truth, no noise
                label_leak = 2 if meta.key == 1 else 1
            else:
                # eval => from CSV if found, else 0
                fn = meta.file_name
                if fn in self.llm_map:
                    val = self.llm_map[fn].lower().strip()
                    if val == 'spoof':
                        label_leak = 1
                    elif val == 'bonafide':
                        label_leak = 2
                    else:
                        label_leak = 0
                else:
                    label_leak = 0

            self.data_llm.append(label_leak)

        self.length = len(self.data_x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        x => raw wave
        y => ground truth 0/1
        meta => ASVFile
        llm_label => 0,1,2
        """
        x = self.data_x[idx]
        y = self.data_y[idx]  # 0=spoof,1=bonafide
        meta = self.files_meta[idx]
        llm_label = self.data_llm[idx]
        return x, y, meta, llm_label

    def parse_llm_csv(self, csv_path):
        """
        CSV lines:  file_name,label
          e.g. "spoof_Beth_Behrs_audio_93,spoof"
        """
        d = {}
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            r = csv.reader(f)
            for row in r:
                if len(row) < 2:
                    continue
                fname = row[0].strip()
                lab = row[1].strip()
                d[fname] = lab
        return d

    def parse_protocols_file(self, protocols_fname):
        lines = open(protocols_fname).read().strip().split('\n')
        files_meta = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            asv_obj = self._parse_line(line)
            files_meta.append(asv_obj)
        return files_meta

    def read_file(self, meta):
        data_x, sr = sf.read(meta.path)
        return data_x, float(meta.key), meta.sys_id

    def _parse_line(self, line):
        """
        Format: "<speaker_id> <file_name> <track> <sys_id> <key>"
        Example: "Beth spoof_Beth_Behrs_audio_93 - A09 spoof"
        """
        toks = line.split()
        spkid = toks[0]
        file_name = toks[1]
        sys_id_str = toks[3]
        key_str = toks[4].lower()
        # 1=bonafide, 0=spoof
        key_val = 1 if key_str == 'bonafide' else 0

        path = os.path.join(self.files_dir, file_name + '.flac')
        # Map sys_id
        if sys_id_str in self.sysid_dict:
            sys_id_val = self.sysid_dict[sys_id_str]
        else:
            sys_id_val = 0

        return ASVFile(spkid, file_name, path, sys_id_val, key_val)
