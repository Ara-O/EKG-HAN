from torch.utils.data import Sampler
from typing import List, Optional
from collections import defaultdict
import random
import math

class LengthBucketBatchSampler(Sampler):
    def __init__(self,
                 lengths: List[int],
                 batch_size: int,
                 drop_last: bool = False):
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.drop_last = drop_last

        # Build mapping num_of_beats -> list of indices
        self._buckets = defaultdict(list)
        for idx, length in enumerate(self.lengths):
            key = int(length)
            self._buckets[key].append(idx)

        # Convert the number of beats to list
        self.bucket_keys = list(self._buckets.keys())


    def __iter__(self):
      # For each epoch, build batches from buckets.
        batches = []
        for key in self.bucket_keys:
                idxs = list(self._buckets[key])
                if len(idxs) < self.batch_size:
                    # skip this bucket if a full batch can't be made from it
                    continue
                
                random.shuffle(idxs)
                    
                # chunk into batches
                for i in range(0, len(idxs), self.batch_size):
                    batch = idxs[i:i + self.batch_size]
                    if len(batch) < self.batch_size and self.drop_last:
                        continue
                    batches.append(batch)

        random.shuffle(batches)
        
        for batch in batches:
            yield batch

    def __len__(self):
        total = 0
        for key in self.bucket_keys:
            n = len(self._buckets[key])
            if n < self.batch_size:
                continue
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += math.ceil(n / self.batch_size)
        return total
    