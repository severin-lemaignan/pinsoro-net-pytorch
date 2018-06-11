import logging
logging.basicConfig(level=logging.DEBUG)

import math 

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pandas as pd

GOALORIENTED="goaloriented"
AIMLESS="aimless"
ADULTSEEKING="adultseeking"
NOPLAY="noplay"

SOLITARY="solitary"
ONLOOKER="onlooker"
PARALLEL="parallel"
ASSOCIATIVE="associative"
COOPERATIVE="cooperative"

PROSOCIAL="prosocial"
ADVERSARIAL="adversarial"
ASSERTIVE="assertive"
FRUSTRATED="frustrated"
PASSIVE="passive"

ALL_ANNOTATIONS = [GOALORIENTED,
                AIMLESS,
                ADULTSEEKING,
                NOPLAY,
                SOLITARY,
                ONLOOKER,
                PARALLEL,
                ASSOCIATIVE,
                COOPERATIVE,
                PROSOCIAL,
                ADVERSARIAL,
                ASSERTIVE,
                FRUSTRATED,
                PASSIVE]


class PInSoRoDataset(Dataset):
    """The PInSoRo dataset."""

    def __init__(self, path, device, chunksize=1000):
        """
        :param path: (string): Path to the csv file with annotations.
        :param chunksize: nb of CSV rows loaded at the same time
        

        """
        self.device=device
        self.path = path

        # read headers + one row of data
        sample_r = pd.read_csv(path, nrows=1)
        # extract data types
        dtypes = sample_r.drop('id',axis=1).dtypes
        # select np.float32 (instead of default float64) for all columns except timestamp
        self.float32dtypes = {key:np.float32 for key, value in dtypes.items() if value == np.float64 and key != 'timestamp'}


        logging.info("Opening CSV file and counting samples...")
        self.nb_samples = sum(1 for line in open(path)) - 1
        logging.info("Found %d samples" % self.nb_samples)

        logging.info("Loading CSV file in chunks (chunksize=%d)" % chunksize)
        self.chunksize = chunksize

        self.len = int(self.nb_samples / self.chunksize)
        self.current_chunk_idx = 0
        self.current_chunk = next(
                                pd.read_csv(self.path, 
                                            skiprows= 1, # skip header
                                            chunksize=self.chunksize, 
                                            dtype=self.float32dtypes)
                            )

    def makeAnnotationTensor(self, constructs):
        """
        The social behaviours of the 2 children is encoded as a single one-hot vector.
        For instance, if purple child is <aimless, solitary, prosocial> and the yellow child is <aimless, onlooker, hostile>, it might be encoded as:
        <0, 1, 0, 0, 1, 0 ............... 1, 0, 0, 0>
         \--------/  \----          ...    --------/
          task engag. social enga.         social attitude
          purple        purple               yellow
        """

        tensor = torch.zeros(1, len(ALL_ANNOTATIONS) * 2,  # x2 -> purple child + yellow child
                             device=self.device)
        #import pdb;pdb.set_trace()

        for c in constructs[0:3]: # purple child construct
            if pd.isnull(c):
                #raise RuntimeError("Missing annotations!")
                logging.warning("Missing annotation -- replaced by 'no construct not present'")
                continue
                
            tensor[0][ALL_ANNOTATIONS.index(c)] = 1

        for c in constructs[3:6]:  # yellow child constructs
            if pd.isnull(c):
                #raise RuntimeError("Missing annotations!")
                logging.warning("Missing annotation -- replaced by 'no construct not present'")
                continue
            tensor[0][ALL_ANNOTATIONS.index(c) + len(ALL_ANNOTATIONS)] = 1

        return tensor

    def __len__(self):
        return self.nb_samples

    def __getitem__(self, idx):


        chunk_idx, chunk_offset = idx // self.chunksize, idx % self.chunksize

        # do we need to fetch the next chunk of our CSV file?
        while chunk_idx > self.current_chunk_idx:
            logging.debug("Fetching next chunk (samples %d to %d)" % (chunk_idx * self.chunksize, (chunk_idx +1)*self.chunksize - 1))

            self.current_chunk = next(
                                pd.read_csv(self.path, 
                                            skiprows=self.current_chunk_idx*self.chunksize + 1, # skip header as well
                                            chunksize=self.chunksize, 
                                            dtype=self.float32dtypes)
                                )

            self.current_chunk_idx += 1

        
        poses_tensor = torch.tensor(
                            self.current_chunk.iloc[chunk_offset, 7:147].astype(np.float32).values,
                            device=self.device, requires_grad=True
                       )
            
        annotations_tensor = self.makeAnnotationTensor(self.current_chunk.iloc[chunk_offset, 432:438])

        return poses_tensor, annotations_tensor

def collate_minibatch(batch):
    transposed = zip(*batch)
    return [torch.stack(samples, 0) for samples in transposed]


def train_validation_loaders(dataset, valid_fraction =0.1, **kwargs):
    """
    Borrowed from https://github.com/ZmeiGorynych/basic_pytorch/blob/master/data_utils/data_sources.py#L24
    """
    # num_workers
    # batch_size
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(math.floor(valid_fraction* num_train))

    if not('shuffle' in kwargs and not kwargs['shuffle']):
            #np.random.seed(random_seed)
            np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(dataset,
                              sampler=train_sampler,
                              collate_fn=collate_minibatch,
                              **kwargs)
    valid_loader = DataLoader(dataset,
                              sampler=valid_sampler,
                              collate_fn=collate_minibatch,
                              **kwargs)

    return train_loader, valid_loader

if __name__ == "__main__":
    import sys


    d = PInSoRoDataset(sys.argv[1], chunksize=1000)
    loader = DataLoader(d, batch_size=10, num_workers=1, shuffle=False)
    print(len(d))
    for batch_idx, data in enumerate(loader):
        print('batch: {}\tdata: {}'.format(batch_idx, data))
    #print(d[0])
