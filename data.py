import logging
logging.basicConfig(level=logging.DEBUG)

import os
import math 
import time

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pandas as pd

DATASET_FILENAME="pinsoro-complete"

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

CONSTRUCT_CLASSES={"task-engagement":TASK_ENGAGEMENT,
                   "social-engagement":SOCIAL_ENGAGEMENT,
                    "social-attitude":SOCIAL_ATTITUDE}

ALL_ANNOTATIONS = TASK_ENGAGEMENT + SOCIAL_ENGAGEMENT + SOCIAL_ATTITUDE

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class PInSoRoDataset(Dataset):
    """The PInSoRo dataset."""

    def __init__(self, path, device, batch_size, seq_size, constructs_class=None):
        """
        :param path: (string): Root of the dataset. Individual (per-recording) datasets are looked up from here.
        :param device: the pytorch device (cpu or cuda)
        :param batch_size: the batch size is used to return a dataset length which is a multiple of it
        :constructs_class: one of ALL_ANNOTATIONS, TASK_ENGAGEMENT, SOCIAL_ENGAGEMENT or SOCIAL_ATTITUDE
        """
        start = time.time()

        self.POSES_INPUT_SIZE = (70 + 18) * 2 * 2 # (face + skel) * (x,y) * (purple, yellow)
        self.POSES_INPUT_IDX = 0
        self.ANNOTATIONS_IDX=self.POSES_INPUT_IDX + self.POSES_INPUT_SIZE + 1

        self.constructs_class = ALL_ANNOTATIONS if constructs_class is None else constructs_class
        self.ANNOTATIONS_OUTPUT_SIZE = len(self.constructs_class) * 2

        self.device=device
        self.batch_size = batch_size
        self.seq_size = seq_size

        self.dtypes = None

        paths = self.lookup_data(path)

        self.dataset = {}

        tot_samples = 0
        # This list holds a long list of tuple (dataset id, index), one per sample
        self.indices = []


        for i, id in enumerate(paths):
            if self.dtypes is None:
                self.generate_dtypes(paths[id])

            logging.info("Loading dataset %d/%d: %s" % (i+1, len(paths), paths[id]))
            self.dataset[id] = pd.read_csv(paths[id], 
                                           skiprows= 1, # skip header
                                           names=self.dtypes.keys(),
                                           dtype=self.dtypes)

            logging.info("Extracting valid samples...")
            tot_samples += len(self.dataset[id])
            #import pdb;pdb.set_trace()
            indices = self.valid_samples_indices(self.dataset[id])

            self.indices += [(id, idx) for idx in indices]


        logging.info("Kept %d samples (%d%% of the total)" % (len(self.indices), len(self.indices) * 100./tot_samples))

        logging.info("Dataset initialization took %s" % timeSince(start))

    def generate_dtypes(self, path):
        # read headers + one row of data
        self.dtypes={}

        sample_r = pd.read_csv(path, nrows=1)
        # extract data types
        orig_dtypes = sample_r.drop('id',axis=1).dtypes
        # select np.float32 (instead of default float64) for all columns except timestamp

        self.dtypes.update({key:np.float32 for key, value in orig_dtypes.items() if "face" in key})
        self.dtypes.update({key:np.float32 for key, value in orig_dtypes.items() if "skel" in key})

        self.dtypes.update({key:np.object for key in orig_dtypes.keys()[431:438]})


    def missing_annotations(self, data):
        """ Returns a 1-dimensional numpy array matching the num of rows in `data`, where True means that
        the annotations are missing for this specific sample.
        """

        if self.constructs_class == TASK_ENGAGEMENT:
            return data[['purple_child_task_engagement', 'yellow_child_task_engagement']].isnull().values.any(axis=1)

        if self.constructs_class == SOCIAL_ENGAGEMENT:
            return data[['purple_child_social_engagement', 'yellow_child_social_engagement']].isnull().values.any(axis=1)

        if self.constructs_class == SOCIAL_ATTITUDE:
            return data[['purple_child_social_attitude', 'yellow_child_social_attitude']].isnull().values.any(axis=1)

        if self.constructs_class == ALL_ANNOTATIONS:
            return data[['purple_child_task_engagement', 'purple_child_social_engagement',
                            'purple_child_social_attitude', 'yellow_child_task_engagement',
                            'yellow_child_social_engagement', 'yellow_child_social_attitude']].isnull().values.any(axis=1)

    def valid_samples_indices(self, data):
        """
        Valid sample indices are indices with at least seq_size samples
        following them, and the *last* sample of the sequence must have
        annotations.

        """

        indices = []

        idx=0
        #import pdb;pdb.set_trace()
        missing_anns = self.missing_annotations(data)

        for missing in missing_anns[self.seq_size:]:
            if not missing:
                indices.append(idx)
            idx+=1

        return indices


    def lookup_data(self, root):

        paths = {}

        logging.info("Looking for recordings...")

        for dirpath, dirs, files in os.walk(root, topdown=False):
            for name in files:
                fullpath = os.path.join(dirpath, name)
                if name.startswith(DATASET_FILENAME) and name.endswith("csv"):
                    if "no-annotations" in name:
                        logging.warning("%s is missing annotations. Skipping it." % fullpath)
                        continue
                    id="%s-%s" % (dirpath.split(os.sep)[-1], name[:-4].split("-")[-1]) # id is recording date (ie folder name) followed by annotator name
                    paths[id] = fullpath

        logging.info("Found %d datasets (child-child condition only) with annotations" % len(paths))

        return paths


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
        return len(self.indices)

    def fill_NaN_with_unif_rand(self, a):
        m = np.isnan(a) # mask of NaNs
        a[m] = np.random.uniform(0., 1.)
        return a

    def __getitem__(self, idx):


        id, indice = self.indices[idx]

        #import pdb;pdb.set_trace()
        poses_np = self.dataset[id].iloc[indice:indice + self.seq_size, self.POSES_INPUT_IDX:self.POSES_INPUT_IDX+self.POSES_INPUT_SIZE].astype(np.float32).values
        poses_np = self.fill_NaN_with_unif_rand(poses_np)

        poses_tensor = torch.tensor(
                            poses_np,
                            requires_grad=True
                       )
            
        # the annotations we need are the *last* in the returned sequence
        annotations_tensor = self.makeAnnotationTensor(
                        self.dataset[id].iloc[
                                indice + self.seq_size-1, 
                                self.ANNOTATIONS_IDX:self.ANNOTATIONS_IDX+6], 
                        self.constructs_class)

        return poses_tensor, annotations_tensor

def collate_minibatch(batch):
    transposed = zip(*batch)
    return [torch.stack(samples, 0) for samples in transposed]


def train_validation_loaders(dataset, valid_fraction=0.2, randomize_split=True, **kwargs):
    """
    Returns 2 dataloaders, one for training, one for validation.
    Samples are returned in random order.

    Based on https://github.com/ZmeiGorynych/basic_pytorch/blob/master/data_utils/data_sources.py#L24
    """
    # num_workers
    # batch_size
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(math.floor(valid_fraction*num_train))

    if randomize_split:
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

    #device = torch.device("cuda") 
    device = torch.device("cpu") 


    d = PInSoRoDataset(path=sys.argv[1], 
                       batch_size=1,
                       seq_size=300,
                       constructs_class=SOCIAL_ENGAGEMENT,
                       device=device)
    print(d[10])
    #loader = DataLoader(d, batch_size=1, num_workers=1, shuffle=False)
    #print(len(d))
    #for batch_idx, data in enumerate(loader):
    #    print('batch: {}\tdata: {}'.format(batch_idx, data))
    #print(d[0])
