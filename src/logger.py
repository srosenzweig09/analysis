from traceback import extract_stack
from colors import H, W, FAIL

for x in extract_stack():
    if not x[0].startswith('<frozen importlib'):
        filename = x[0]
        break

def info(string):
    print(f"-- [INFO] -- {H}{filename}{W} -- " + string)
def error(string):
    print(f"!! [{FAIL}ERROR{W}] !! {H}{filename}{W} -- " + string)


