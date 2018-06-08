import pandas as pd

class Read(object):

    read_data = {}
    """
        file = full path and file name that will be read.
    """
    def read_csv(file, sep=',', delimiter):

        if read_data.get(file) == None:
            data = pd.read_csv(filepath_or_buffer=file,sep=sep,delimiter=delimiter)
            read_data[file]=data
        else:
            data = read_data.get(file)
        
        return data