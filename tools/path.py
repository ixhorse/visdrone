"""
set data path
"""

import os
user_home = os.path.expanduser('~')

class Path(object):
    @staticmethod
    def db_root_dir():
        return os.path.join(user_home, 'data/visdrone2019')