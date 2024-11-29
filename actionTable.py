import itertools
import numpy as np

class ActionTable:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # Generate all possible combinations
        base_range = range(input_size)
        self.combinations = list(itertools.product(base_range, repeat=output_size))
        self.combinations = [c for c in self.combinations if len(set(c)) == len(c)]
        self.table_size = len(self.combinations)
    
    def __len__(self):
        return self.table_size
    
    def get_action(self, index):
        if index < 0 or index >= self.table_size:
            raise IndexError(f"Index {index} out of range [0, {self.table_size-1}]")
        return np.array(self.combinations[index])

    def __getitem__(self, index):
        return self.get_action(index)