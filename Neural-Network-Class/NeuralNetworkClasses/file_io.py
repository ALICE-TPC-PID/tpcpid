import string
import uproot
import uproot3
import numpy as np

class write_to_file():
    
    def __init__(self,labels,data):
        super().__init__()
        self.labels = labels
        self.data = data
    
    def write(self, filename="out.txt"):
        output_data = np.vstack((np.array([self.labels]), np.array(self.data)))
        np.savetxt(filename, output_data, fmt='%s')
        print("Data saved to file", filename)