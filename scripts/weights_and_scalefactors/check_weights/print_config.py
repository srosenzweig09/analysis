from argparse import ArgumentParser
from configparser import ConfigParser

parser = ArgumentParser()
parser.add_argument("--config", default="config/feynnet.cfg")
args = parser.parse_args()

config = ConfigParser()
config.read(args.config)

model_version = config['feynnet']['version']
model_name = config['feynnet']['name']
model_path = config['feynnet']['path'].replace('version', model_name)

print("Model Configuration:")
print(f"  tmp file: {args.config}")
print(f"  version:  {model_version}")
print(f"  name:     {model_name}")
print(f"  path:     {model_path}")
print()