class DataHolder:
    """
    Class to store all the data information
    """
    def __init__(self,
                 train_dataset,
                 train_labels,
                 valid_dataset,
                 valid_labels,
                 test_dataset,
                 test_labels):
        """
        :type train_dataset: numpy array
        :type train_labels: numpy array
        :type valid_dataset: numpy array
        :type valid_labels: numpy array
        :type test_dataset: numpy array
        :type test_labels: numpy array
        """
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.valid_dataset = valid_dataset
        self.valid_labels = valid_labels
        self.test_dataset = test_dataset
        self.test_labels = test_labels
