import tensorflow as tf
import sys, argparse


parser = argparse.ArgumentParser(description='Get the type of a flower.')
parser.add_argument('image', type=str, help="The image path")
parser.add_argument('model', type=str, help="your model path")
parser.add_argument('--top_k', help='Return the top K most likely classes:', type=int, default=5)
parser.add_argument('--category_names', help='Path to a JSON file mapping labels to flower names:', type=str)
args = parser.parse_args()


# parser.add_argument('--top_k', help='Return the top K most likely classes:', type=int, default=5)
# parser.add_argument('--category_names', help='Path to a JSON file mapping labels to flower names:', type=str)

if __name__ == '__main__':
    print(args.image, args.model, args.top_k, args.category_names)
