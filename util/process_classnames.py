# ------------------------------------------------------------------------
# Open World Object Detection in the Era of Foundation Models
# Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang
# ------------------------------------------------------------------------

import argparse
import os 
import pandas as pd 
import numpy as np

from process_classnames_utils import get_first_item_from_json,read_file,save_to_file

def main(args):

    ## Special case
    if args.coco == True:
        a = ("t4_owod_known.txt"   , "t1_owod_known.txt"   , "t1_owod_unknown_classnames_groundtruth.txt")
        b = ("t4_owod_known.txt"   , "t2_owod_known.txt"   , "t2_owod_unknown_classnames_groundtruth.txt")
        c = ("t4_owod_known.txt"   , "t3_owod_known.txt"   , "t3_owod_unknown_classnames_groundtruth.txt")
        d = ("t4_owdetr_known.txt" , "t1_owdetr_known.txt" , "t1_owdetr_unknown_classnames_groundtruth.txt")
        e = ("t4_owdetr_known.txt" , "t2_owdetr_known.txt" , "t2_owdetr_unknown_classnames_groundtruth.txt") 
        f = ("t4_owdetr_known.txt" , "t3_owdetr_known.txt" , "t3_owdetr_unknown_classnames_groundtruth.txt")

        path          = os.path.join(args.data_path ,"COCO")

        for files in [a,b,c,d,e,f]:
            all_classes_labels   = set(read_file(os.path.join(path ,files[0])))
            known_classes_labels = set(read_file(os.path.join(path ,files[1])))
            diff                 = all_classes_labels - known_classes_labels

            save_path            =  os.path.join(path ,files[2])
            save_to_file(iterable= diff, file_path=save_path)


    for dataset in args.Datasets:
        path          = os.path.join(args.data_path ,dataset)
        all_classes   = os.path.join(path ,args.class_names)
        known_classes = os.path.join(path ,args.known_classnames)

        all_classes_labels   = set(read_file(all_classes))
        known_classes_labels = set(read_file(known_classes))
        diff                 = all_classes_labels   -known_classes_labels

        save_path            =  os.path.join(path ,args.output_path)
        save_to_file(iterable= diff, file_path=save_path)

 

       

        #import pdb;pdb.set_trace()
    


if __name__ == "__main__":

    default_datasets         = ["AQUA", "DIOR_FIN", "SYNTH", "XRAY", "NEUROSURGICAL_TOOLS_FIN",]
    default_class_names      = "classnames.txt"
    default_known_classnames = "known_classnames.txt"
    default_ouptut           = "unknown_classnames_ground_truth.txt"
    default_path             = os.path.join("..", "data", "RWOD","ImageSets")
    
    parser = argparse.ArgumentParser(description="Process parameters with default values")
    parser.add_argument("--Datasets",           type=list, default=default_datasets,           help="List of datasets")
    parser.add_argument("--class_names",        type=str,  default=default_class_names ,       help="classname text path")
    parser.add_argument("--known_classnames",   type=str,  default= default_known_classnames , help="unkwon classnames text path")
    parser.add_argument("--data_path",          type=str,  default= default_path ,             help="Default Dataset path")
    parser.add_argument("--output_path",        type=str,  default= default_ouptut  ,          help="Default output")
    parser.add_argument("--coco", action="store_true", help="Default output")
    args = parser.parse_args()
    main(args)
