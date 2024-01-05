# ------------------------------------------------------------------------
# Open World Object Detection in the Era of Foundation Models
# Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang
# ------------------------------------------------------------------------

import argparse
import os 
import pandas as pd 
import numpy as np
from process_classnames_utils import get_first_item_from_txt


def main(args):

    image_net_labels = get_first_item_from_txt(args.ImageNetLabelsPath)
    

if __name__ == "__main__":

    default_datasets     = ["AQUA", "DIOR_FIN", "SYNTH", "XRAY", "NEUROSURGICAL_TOOLS_FIN",]
    default_experiments  = ["no_refinement","no_refinement-no_att_selection","no_refinement-no_att_selection-no_adapt"]

   
    default_extension    = os.path.join("..", "data", "RWOD","ImageSets","IMAGE_NET","imagenet1000_clsidx_to_labels.txt")
    parser = argparse.ArgumentParser(description="Process parameters with default values")
    parser.add_argument("--Datasets",          type=list, default=default_datasets,    help="List of datasets")
    parser.add_argument("--Experiments",       type=list, default=default_experiments, help="List of experiments")
    parser.add_argument("--ImageNetLabelsPath",type=str,  default=default_extension,   help="File extension")

    args = parser.parse_args()
    main(args)
