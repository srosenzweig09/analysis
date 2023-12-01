import subprocess, shlex

def check_output(cmd, clean_list=True):
    output = subprocess.check_output(shlex.split(cmd))
    output = output.decode('UTF-8').split('\n')
    if clean_list: output = [out for out in output if 'analysis_tar' not in out]
    if clean_list: output = [out for out in output if out != '']
    return output

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout