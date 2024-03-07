import os

with open("filelists/Summer2018UL/central.txt") as f:
    filelist = f.readlines()

for file in filelist:
    file = file.strip('\n')
    if not os.path.isfile(file): print(file)