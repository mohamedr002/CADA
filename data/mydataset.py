import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, labels):
        """Reads source and target sequences from processing file ."""
        self.input_tensor = (torch.from_numpy(data)).float()

        self.label = (torch.torch.FloatTensor(labels))
        self.num_total_seqs = len(self.input_tensor)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        input_seq = self.input_tensor[index]
        input_labels = self.label[index]
        return input_seq, input_labels

    def __len__(self):
        return self.num_total_seqs


def create_dataset(data, batch_size, shuffle, drop_last):
    trainX, validX, testX, trainY, validY, testY = data
    train_dl = DataLoader(MyDataset(trainX, trainY), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    valid_dl = DataLoader(MyDataset(validX, validY), batch_size=10, shuffle=False, drop_last=False)
    test_dl = DataLoader(MyDataset(testX, testY), batch_size=10, shuffle=False, drop_last=False)
    return train_dl, valid_dl, test_dl


def create_dataset_full(data, batch_size=10, shuffle=True, drop_last=True):
    trainX, testX, trainY, testY = data
    train_dl = DataLoader(MyDataset(trainX, trainY), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    test_dl = DataLoader(MyDataset(testX, testY), batch_size=10, shuffle=False, drop_last=False)
    return train_dl, test_dl
