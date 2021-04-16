from __future__ import absolute_import
from __future__ import print_function

import argparse
import yaml

from mimic3benchmark.mimic3csv import *
from mimic3benchmark.preprocessing import add_hcup_ccs_2015_groups, make_phenotype_label_matrix
from mimic3benchmark.util import dataframe_from_csv

parser = argparse.ArgumentParser(
    description='Extract per-subject data from MIMIC-III CSV files.')
parser.add_argument('mimic3_path', type=str,
                    help='Directory containing MIMIC-III CSV files.')
parser.add_argument('output_path', type=str,
                    help='Directory where per-subject data should be written.')
parser.add_argument('--event_tables', '-e', type=str, nargs='+', help='Tables from which to read events.',
                    default=['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS'])
parser.add_argument('--phenotype_definitions', '-p', type=str,
                    default=os.path.join(os.path.dirname(
                        __file__), '../resources/hcup_ccs_2015_definitions.yaml'),
                    help='YAML file with phenotype definitions.')
parser.add_argument('--itemids_file', '-i', type=str,
                    help='CSV containing list of ITEMIDs to keep.')
parser.add_argument('--verbose', '-v', dest='verbose',
                    action='store_true', help='Verbosity in output')
parser.add_argument('--quiet', '-q', dest='verbose',
                    action='store_false', help='Suspend printing of details')
parser.set_defaults(verbose=True)
parser.add_argument('--test', action='store_true',
                    help='TEST MODE: process only 1000 subjects, 1000000 events.')
args, _ = parser.parse_known_args()

try:
    os.makedirs(args.output_path)
except:
    pass

notes = read_notes_table(args.mimic3_path)  # Added by Bhaskar Rudra
notes.to_csv(os.path.join(args.output_path, 'all_notes.csv'),
             index=False)  # Added by Bhaskar Rudra
