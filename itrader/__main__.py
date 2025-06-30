import fire, sys
from itrader.rl.demo_reinforce import train

def main(cmd="help", *args):
    if cmd == "train":
        train()
    else:
        print("Usage: itrader [train]")
if __name__ == "__main__":
    main(*sys.argv[1:])
