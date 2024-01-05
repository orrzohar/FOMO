# ------------------------------------------------------------------------
# Open World Object Detection in the Era of Foundation Models
# Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang
# ------------------------------------------------------------------------

import argparse
import os
import pandas as pd


def append_rows(files, output_file,method="df"):
    try:

        if method == "df":
        ### Using pandas ###
            merged_data = pd.DataFrame()

            for file in files:
                df = pd.read_csv(file)
                # use concat
                merged_data = pd.concat([merged_data, df], ignore_index=True)
               

                

            print("Saving at: ", output_file)
            merged_data.to_csv(output_file, index=False)

        else:
        ### Using files ###
            with open(output_file, 'w') as output:
                for i,file in enumerate(files):
                    with open(file, 'r') as f:
                        if i == 0:
                            for line in f:
                                output.write(line)
                        else:
                            for j, line in enumerate(f):
                                if j != 0:
                                    output.write(line)


            print(f"Rows from files {', '.join(files)} have been appended to {output_file}")
    except FileNotFoundError:
        print("One or more input files not found.")

    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Append rows from multiple files into one output file.")
    parser.add_argument("--input_file",    default = "owod_large-patch14_N_2.csv",   help="Output file to store the appended rows")
    parser.add_argument("--files_appendix",default=[0,20,40,60], nargs='+',          help="Input files to append rows from")
    parser.add_argument("--output_file",   default=None,                             help="Output file to store the appended rows")
    parser.add_argument("--method",        default="df",                             help="method to merge")
    args  = parser.parse_args()
    files =  [args.input_file.replace("N",f"{i}").replace(",","") for i in args.files_appendix ]
    #print(files)
    if args.output_file is None:
        args.output_file = args.input_file.replace("N","merged").replace(",","") 

    #print(os.listdir("."))
    
    #import pdb; pdb.set_trace()
    append_rows(files, args.output_file,args.method)

if __name__ == "__main__":
    main()
