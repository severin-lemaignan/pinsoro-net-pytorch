import logging
logging.basicConfig(level=logging.DEBUG)

import math 

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pandas as pd

MISSINGDATA="missingdata"

GOALORIENTED="goaloriented"
AIMLESS="aimless"
ADULTSEEKING="adultseeking"
NOPLAY="noplay"

TASK_ENGAGEMENT = [GOALORIENTED,
                   AIMLESS,
                   ADULTSEEKING,
                   NOPLAY]

SOLITARY="solitary"
ONLOOKER="onlooker"
PARALLEL="parallel"
ASSOCIATIVE="associative"
COOPERATIVE="cooperative"

SOCIAL_ENGAGEMENT = [SOLITARY,
                     ONLOOKER,
                     PARALLEL,
                     ASSOCIATIVE,
                     COOPERATIVE]

PROSOCIAL="prosocial"
ADVERSARIAL="adversarial"
ASSERTIVE="assertive"
FRUSTRATED="frustrated"
PASSIVE="passive"

SOCIAL_ATTITUDE = [PROSOCIAL,
                ADVERSARIAL,
                ASSERTIVE,
                FRUSTRATED,
                PASSIVE]

ALL_ANNOTATIONS = TASK_ENGAGEMENT + SOCIAL_ENGAGEMENT + SOCIAL_ATTITUDE

class PInSoRoDataset(Dataset):
    """The PInSoRo dataset."""

    def __init__(self, path, device, constructs_class=None, chunksize=1000):
        """
        :param path: (string): Path to the csv file with annotations.
        :param device: the pytorch device (cpu or cuda)
        :constructs_class: one of ALL_ANNOTATIONS, TASK_ENGAGEMENT, SOCIAL_ENGAGEMENT or SOCIAL_ATTITUDE
        :param chunksize: nb of CSV rows loaded at the same time
        

        """
        self.POSES_INPUT_SIZE = (70 + 18) * 2 * 2 # (face + skel) * (x,y) * (purple, yellow)
        self.POSES_INPUT_IDX = 0
        self.ANNOTATIONS_IDX=self.POSES_INPUT_IDX + self.POSES_INPUT_SIZE + 1

        self.constructs_class = ALL_ANNOTATIONS if constructs_class is None else constructs_class
        self.ANNOTATIONS_OUTPUT_SIZE = len(self.constructs_class) * 2

        self.device=device
        self.path = path

        # read headers + one row of data
        sample_r = pd.read_csv(path, nrows=1)
        # extract data types
        orig_dtypes = sample_r.drop('id',axis=1).dtypes
        # select np.float32 (instead of default float64) for all columns except timestamp
        self.dtypes = {}

        self.dtypes.update({key:np.float32 for key, value in orig_dtypes.items() if "face" in key})
        self.dtypes.update({key:np.float32 for key, value in orig_dtypes.items() if "skel" in key})

        self.dtypes.update({key:np.object for key in orig_dtypes.keys()[431:438]})


        logging.info("Opening CSV file and counting samples...")
        self.nb_samples = sum(1 for line in open(path)) - 1
        logging.info("Found %d samples" % self.nb_samples)

        logging.info("Loading CSV file in chunks (chunksize=%d)" % chunksize)
        self.chunksize = chunksize

        self.len = int(self.nb_samples / self.chunksize)
        self.current_chunk_idx = 0
        #import pdb;pdb.set_trace()
        self.current_chunk = next(
                                pd.read_csv(self.path, 
                                            skiprows= 1, # skip header
                                            chunksize=self.chunksize, 
                                            names=self.dtypes.keys(),
                                            dtype=self.dtypes)
                            )

        #import pdb;pdb.set_trace()

    def makeAnnotationTensor(self, constructs, annotation_set=None):
        """
        The social behaviours of the 2 children is encoded as a single one-hot vector.
        For instance, if purple child is <aimless, solitary, prosocial> and the yellow child is <aimless, onlooker, hostile>, it might be encoded as:
        <0, 1, 0, 0, 1, 0 ............... 1, 0, 0, 0>
         \--------/  \----          ...    --------/
          task engag. social enga.         social attitude
          purple        purple               yellow
        """

        if annotation_set is None:
            annotation_set = ALL_ANNOTATIONS

        tensor = torch.zeros(len(annotation_set) * 2)  # x2 -> purple child + yellow child

        #import pdb;pdb.set_trace()

        for c in constructs[0:3]: # purple child construct
            if pd.isnull(c):
                #raise RuntimeError("Missing annotations!")
                logging.debug("Missing annotation -- replaced by 'construct not present'")
                continue
                
            if c in annotation_set:
                tensor[annotation_set.index(c)] = 1

        for c in constructs[3:6]:  # yellow child constructs
            if pd.isnull(c):
                #raise RuntimeError("Missing annotations!")
                logging.debug("Missing annotation -- replaced by 'construct not present'")
                continue

            if c in annotation_set:
                tensor[annotation_set.index(c) + len(annotation_set)] = 1

        return tensor

    def __len__(self):
        return self.nb_samples

    def fill_NaN_with_unif_rand(self, a):
        m = np.isnan(a) # mask of NaNs
        a[m] = np.random.uniform(0., 1.)
        return a

    def __getitem__(self, idx):


        chunk_idx, chunk_offset = idx // self.chunksize, idx % self.chunksize

        # do we need to fetch the next chunk of our CSV file?
        while chunk_idx > self.current_chunk_idx:
            logging.debug("Fetching next chunk (samples %d to %d)" % (chunk_idx * self.chunksize, (chunk_idx +1)*self.chunksize - 1))

            self.current_chunk = next(
                                pd.read_csv(self.path, 
                                            skiprows=self.current_chunk_idx*self.chunksize + 1, # skip header as well
                                            chunksize=self.chunksize,
                                            names=self.dtypes.keys(),
                                            dtype=self.dtypes)
                                )

            self.current_chunk_idx += 1

        poses_np = self.current_chunk.iloc[chunk_offset, self.POSES_INPUT_IDX:self.POSES_INPUT_IDX+self.POSES_INPUT_SIZE].astype(np.float32).values
        #poses_np = self.fill_NaN_with_unif_rand(poses_np)

        poses_tensor = torch.tensor(
                            poses_np,
                            requires_grad=True
                       )
            
        annotations_tensor = self.makeAnnotationTensor(self.current_chunk.iloc[chunk_offset, self.ANNOTATIONS_IDX:self.ANNOTATIONS_IDX+6], self.constructs_class)

        #import pdb;pdb.set_trace()
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

    device = torch.device("cuda") 
    #device = torch.device("cpu") 


    d = PInSoRoDataset(sys.argv[1], chunksize=1000, device=device)
    print(d[10])
    loader = DataLoader(d, batch_size=1, num_workers=1, shuffle=False)
    print(len(d))
    for batch_idx, data in enumerate(loader):
        print('batch: {}\tdata: {}'.format(batch_idx, data))
    #print(d[0])
