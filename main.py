#!/usr/bin/env python3

import argparse

from src import split
from src import fingerprint
from src import top_n
from src import testing

if __name__== "__main__":
    def restricted_float(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
        return x

    #Parse args
    parser = argparse.ArgumentParser(description='Analyze a GAN for patterns.')
    subparsers = parser.add_subparsers()

    p = subparsers.add_parser( "split", description='Generate the fingerprint and testing splits for a set of datasets.', formatter_class=argparse.ArgumentDefaultsHelpFormatter, help="Generate splits.")
    p.add_argument('--fingerprint', default=1000, help='Number of images to use for the fingerprint', type=int)
    p.add_argument('--testing', default=1000, help='Number of images to use in testing', type=int)
    p.add_argument('--output_root', default='results', help='Root output directory')
    p.add_argument('--split', default='split', help='Sub path to the split filenames directory')
    p.add_argument('--no-allow-upsample', dest='allow_upsample', action='store_true', help='Only use images which are larger than the resolution in datasets.json')
    p.set_defaults(allow_upsample=False)
    p.add_argument('--overwrite', dest='overwrite', action='store_true', help='Overwrite previous results.')
    p.set_defaults(allow_upsample=False)
    p.set_defaults (func=split.process_split)

    p = subparsers.add_parser( "fingerprint", description='Generate the fingerprints for the datasets in datasets.json.', formatter_class=argparse.ArgumentDefaultsHelpFormatter, help="Generate the fingerprints.")
    p.add_argument('--output_root', default='results', help='Root output directory')
    p.add_argument('--split', default='split', help='Sub path to the split filenames directory')
    p.add_argument('--fingerprint', default='fingerprint', help='Sub path to the fingerprint directory')
    p.add_argument('--overwrite', dest='overwrite', action='store_true', help='Overwrite previous results.')
    p.set_defaults(allow_upsample=False)
    p.set_defaults (func=fingerprint.process_fingerprint)

    p = subparsers.add_parser( "testing", description='Perform the testing for the datasets in datasets.json.', formatter_class=argparse.ArgumentDefaultsHelpFormatter, help="Perform the actual testing.")
    p.add_argument('--output_root', default='results', help='Root output directory')
    p.add_argument('--split', default='split', help='Sub path to the split filenames directory')
    p.add_argument('--fingerprint', default='fingerprint', help='Sub path to the fingerprint directory')
    p.add_argument('--predictions', default='predictions', help='Sub path to the predictions directory')
    p.add_argument('--correlations', default='correlations', help='Sub path to the correlations results directory')
    p.add_argument('--temp', default='temp', help='Sub path to the temp evasion image directory')
    p.add_argument('--test_cases', default="both", choices=["both", "all_to_all", "one_to_one"], help='Which test cases to try. All-to-all compares all datasets with all fingerprints. \
                                                                                                       One-to-one compares each fingerprint with the true dataset and the training dataset. \
                                                                                                       Both covers all-to-all and one-to-one.', type=str)
    p.add_argument('--evasion_type', default="none", choices=["none", "noise", "blur", "jpeg"], help='What type of evasion to perform. See the README for more information.', type=str)
    p.add_argument('--evasion_level', default=0, help='The evasion level. Meaning depends on --evasion_type. See README for more information.', type=float)
    p.add_argument('--threshold', default=None, help='The threshold to use for prediction. Otherwise the threshold is chosen for optimum accuracy for each test case.', type=float)
    p.add_argument('--overwrite', dest='overwrite', action='store_true', help='Overwrite previous results.')
    p.set_defaults(allow_upsample=False)
    p.set_defaults (func=testing.process_testing)

    def bad_args(arguments):
        parser.print_help()
        exit(0)
    parser.set_defaults(func=bad_args)

    arguments = parser.parse_args()
    arguments.func(arguments)
