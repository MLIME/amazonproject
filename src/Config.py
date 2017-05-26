from tools.util import get_log_path


class Config():
    """
    Class to hold all model hyperparams.
    """
    def __init__(self,
                 batch_size=230,
                 patch_size=7,
                 image_size=28,
                 num_labels=17,
                 num_channels=3,
                 num_filters_1=11,
                 num_filters_2=22,
                 hidden_nodes_1=120,
                 hidden_nodes_2=80,
                 hidden_nodes_3=40,
                 learning_rate=0.95,
                 steps_for_decay=150,
                 decay_rate=0.96,
                 dropout=0.75,
                 mean=0,
                 std=0.1,
                 tunning=False,
                 log_path=None):
        """
        :type batch_size: int
        :type patch_size: int
        :type image_size: int
        :type num_channels: int
        :type num_filters_1: int
        :type num_filters_2: int
        :type hidden_nodes_1: int
        :type hidden_nodes_2: int
        :type hidden_nodes_3: int
        :type learning_rate: float
        :type steps_for_decay: float
        :type decay_rate: float
        :type dropout: float
        :type mean: float
        :type std: float
        :type tunning: boolean
        """
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_labels = num_labels
        self.num_channels = num_channels
        self.num_filters_1 = num_filters_1
        self.num_filters_2 = num_filters_2
        self.hidden_nodes_1 = hidden_nodes_1
        self.hidden_nodes_2 = hidden_nodes_2
        self.hidden_nodes_3 = hidden_nodes_3
        self.learning_rate = learning_rate
        self.steps_for_decay = steps_for_decay
        self.decay_rate = decay_rate
        self.dropout = dropout
        self.mean = mean
        self.std = std
        self.tunning = tunning
        if log_path is None:
            self.log_path = get_log_path()
        else:
            self.log_path = log_path
