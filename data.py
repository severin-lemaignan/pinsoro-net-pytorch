import torch
from torch.utils.data import Dataset
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

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        print("Loading %s... this may take a while!" % csv_file)
        self.pinsoro = pd.read_csv(csv_file)


    def makeAnnotationTensor(self, purple_constructs, yellow_constructs):
        """
        The social behaviours of the 2 children is encoded as a single one-hot vector.
        For instance, if purple child is <aimless, solitary, prosocial> and the yellow child is <aimless, onlooker, hostile>, it might be encoded as:
        <0, 1, 0, 0, 1, 0 ............... 1, 0, 0, 0>
         \--------/  \----          ...    --------/
          task engag. social enga.         social attitude
          purple        purple               yellow
        """

        tensor = torch.zeros(1, len(ALL_ANNOTATIONS) * 2) # x2 -> purple child + yellow child

        for c in purple_constructs:
            if pd.isnull(c):
                return None
            tensor[0][ALL_ANNOTATIONS.index(c)] = 1

        for c in yellow_constructs:
            if pd.isnull(c):
                return None
            tensor[0][ALL_ANNOTATIONS.index(c) + len(ALL_ANNOTATIONS)] = 1

    def __len__(self):
        return len(self.pinsoro)

    def __getitem__(self, idx):

        ann = self.makeAnnotationTensor(self.pinsoro.iloc[idx,432:435], self.pinsoro.iloc[idx,435:438])

        #sample = {'poses': self.pinsoro[idx, 7:147].as_matrix(), 'annotations': ann}
        sample = {'annotations': ann}

        return sample

if __name__ == "__main__":
    import sys
    d = PInSoRoDataset(sys.argv[1])
    print(len(d))
    print(d[0])
    print(d[50])
    print(d[1000])
