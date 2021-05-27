import os
import re
import pandas as pd
import argparse

def main(args):
    mapping = pd.DataFrame()
    # get a list of all directories in jsut dataset root directory
    subdirs = next(os.walk(args.jsut_dir))[1]
    subdirs.sort()
    
    for sd in subdirs:
        wavs = os.listdir(os.path.join(args.jsut_dir, sd, 'wav'))
        # keep first 3 characters and all numbers for audio file names
        for w in wavs:
            w_ = re.sub('[^0-9]', '', w)
            mapping = mapping.append([[w, sd[:3]+w_]])
    
    mapping.to_csv(os.path.join(args.md_dir, 'jsut_audio_naming.csv'), 
                   index=None, header=None)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsut_dir', type=str, required=True,
                        help='Absolute path of the JVS dataset directory')
    parser.add_argument('--md_dir', type=str, required=True,
                        help='Absolute path to metadata directory')
    args = parser.parse_args()
    main(args)
