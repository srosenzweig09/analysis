from traceback import extract_stack

for x in extract_stack():
    if not x[0].startswith('<frozen importlib'):
        filename = x[0]
        break

def info(string):
    print(f"-- [INFO] -- {filename} -- " + string)
def error(string):
    print(f"!! [ERROR] !! {filename} -- " + string)


