import os
import argparse
import signal

def main(args):
    subdirectories = os.listdir(args.input_dir)

    for subdirectory in subdirectories:
        dataset_path = os.path.join(args.input_dir, subdirectory, subdirectory)
        if not os.path.isdir(dataset_path):
            continue
        print(subdirectory)
        
        try:
            # run prepare_normals.py --input_dir {dataset_path}
            os.system(f"python prepare_normals.py --input {dataset_path}")
        except KeyboardInterrupt:
            print("Keyboard interrupt received, terminating the process.")
            exit()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Prepare normals for face images')
    parser.add_argument('--input_dir', type=str, help='Input image path')

    args = parser.parse_args()

    main(args)