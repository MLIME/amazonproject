from datetime import datetime

def get_timestamp(self):
    return datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
