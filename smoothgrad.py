from train.smoothgrad import main
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("id", type=int)
args = parser.parse_args()

main(args.id)
