import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('domain_name', default='CartPole')
    parser.add_argument()