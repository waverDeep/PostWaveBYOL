import copy
from torch.utils.data import Dataset
import torchaudio
import torch


class WaveformDataset(Dataset):
    def __init__(self, data_list):
        super(WaveformDataset, self).__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        waveform01, sample_rate = torchaudio.load(self.data_list[index])
        waveform02 = copy.deepcopy(waveform01)
        # augmentation이 적용되어야 함
        return waveform01, waveform02

def _collate_fn(batch):
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    waveform_lengths = [data[0].size(1) for data in batch]
    max_waveform_length = max(waveform_lengths)
    batch_size = len(batch)

    waveform01_pack = torch.zeros(batch_size, 1, max_waveform_length)
    waveform02_pack = torch.zeros(batch_size, 1, max_waveform_length)

    for x in range(batch_size):
        sample = batch[x]
        waveform01 = sample[0]
        waveform02 = sample[1]
        waveform01_length = waveform01.size(1)
        waveform02_length = waveform02.size(1)
        waveform01_pack[x].narrow(1, 0, waveform01_length).copy_(waveform01)
        waveform02_pack[x].narrow(1, 0, waveform02_length).copy_(waveform02)

    waveform_lengths = torch.IntTensor()
    return waveform01_pack, waveform02_pack, waveform_lengths








