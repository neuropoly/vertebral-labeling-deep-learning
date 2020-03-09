import sys
import os
import argparse
sys.path.insert(0, '/home/GRAMES.POLYMTL.CA/luroub/luroub_local/lurou_local/sct/sct/')
from spinalcordtoolbox.cropping import ImageCropper, BoundingBox
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils import Metavar, SmartFormatter
import sct_utils as sct

import torch
from models import *
from test import *
from Data2array import *
import numpy as np
sys.path.insert(0, '/home/GRAMES.POLYMTL.CA/luroub/luroub_local/lurou_local/sct/sct/')
import nibabel as nib


def get_parser():
    # Mandatory arguments
    parser = argparse.ArgumentParser(
        description="tools to detect and label vertebrae with countception deep learning network ",
        epilog="EXAMPLES:\n",
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip('.py'))

    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        '-i',
        required=True,
        help="Input image. Example: t2.nii.gz",
        metavar=Metavar.file,
    )
    mandatoryArguments.add_argument(
        '-c',
        required=True,
        help="contrast",
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        help="Show this help message and exit")
    optional.add_argument(
        '-t',
        help="threshold",
        metavar=Metavar.str,
    )
    optional.add_argument(
        '-o',
        help="name",
        metavar=Metavar.file,
    )

    return parser


def main(args=None):
    """
    Main function
    :param args:
    :return:
    """
    # get parser args
    if args is None:
        args = None if sys.argv[1:] else ['--help']
    parser = get_parser()
    arguments = parser.parse_args(args=args)
    Im_input = Image(arguments.i)
    contrast = arguments.c

    global cuda_available
    cuda_available = torch.cuda.is_available()

    model = ModelCountception_v2(inplanes=1, outplanes=1)
    if cuda_available:
        model = model.cuda()
        model = model.double()

    if contrast == 't1':
        model.load_state_dict(torch.load('checkpoints/Countception_L2T1.model')['model_weights'])

    elif contrast == 't2':
        model.load_state_dict(torch.load('checkpoints/Countception_L2T2.model')['model_weights'])

    else:
        sct.printv('Error...unknown contrast. please select between t2 and t1.')
        return 100
    sct.printv('retrieving input...')
    Im_input.change_orientation('RPI')
    arr = np.array(Im_input.data)
    #debugging

    sct.printv(arr.shape)
    ind = int(np.round(arr.shape[0] / 2))
    inp = np.expand_dims(np.mean(arr[ind - 2:ind + 2, :, :], 0),-1)
    sct.printv('Predicting coordinate')
    
    coord = prediction_coordinates(inp, model, [0,0], 0, test=False)
    mask_out = np.zeros(arr.shape)
    if len(coord) < 2:
        sct.printv('Error did not work at all, you can try with a different threshold')

    for x in coord:
        mask_out[ind, x[1], x[0]] = 10
    sct.printv('saving image')
    imsh=arr.shape
    to_save = Image(param=[imsh[0],imsh[1],imsh[2]],hdr=Im_input.header)
    to_save.data = mask_out
    if arguments.o is not None:
        to_save.save(arguments.o)
    else:
        to_save.save('labels_first_try.nii')

if __name__ == "__main__":
    main()
