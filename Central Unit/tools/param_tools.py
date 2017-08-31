"""
tools for reading and saving parameters file
"""
from json import dump,load
from os.path import exists, abspath
import logging

def read_param(path='def_param.json'):
    if exists(path):
        logging.info('import parameters [%s]'%abspath(path))
        return load(open(path,'rt'))
    else:
        logging.warning('parameters file do not exist [%s]'%abspath(path))
        return None

def save_param(param,path='def_param.json'):
    logging.info('saving parameters [%s]'%abspath(path))
    dump(param,open(path,'wt'),sort_keys=True,separators=(',',':'),indent=2)

if __name__ == '__main__':
    def_param = {'a':5,'b':1.25,'c':'qwerty'}

    param = read_param()
    if param is None:
        param = def_param

    save_param(param)
