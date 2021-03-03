# !/usr/bin/env python
import glob
import os

import numpy as np
import pytest
from moto import mock_dynamodb2
from moto import mock_s3
from moto import mock_ssm
# our module(s)
from ometiff_processor import OMETIFFProcessor, OMETIFFOutputFile

from base_processor.tests import init_ssm
from base_processor.tests import setup_processor

test_processor_data = [
    'color.tif',
    '4D-series.ome.tiff'
]


@pytest.mark.parametrize("filename", test_processor_data)
def test_explode_assets(filename):
    print "~" * 60
    print " Using test file %s to test exploding PNG assets " % filename
    print "~" * 60

    mock_dynamodb2().start()
    mock_ssm().start()
    mock_s3().start()

    init_ssm()

    # init task
    inputs = {'file': os.path.join('/test-resources', filename)}
    task = OMETIFFProcessor(inputs=inputs)

    setup_processor(task)

    # Load image
    output_file = OMETIFFOutputFile()
    output_file.file_path = task.file
    output_file.load_and_save_assets(output_file.file_path, i_xyzct=(-1, -1, -1, -1, -1), n_xyzct=(-1, -1, -1, -1, -1),
                                     asset_format='png')

    # Since we know first two dimensions are are X and Y
    assert output_file.img_dimensions['dimensions'][0]['assignment'] == 'X'
    assert output_file.img_dimensions['dimensions'][1]['assignment'] == 'Y'

    # Get number of files generated
    num_png_files = len(glob.glob('%s-zoomed/*.png' % os.path.basename(output_file.file_path)))
    print glob.glob('%s-zoomed/*.png' % os.path.basename(output_file.file_path))

    # Ensure correct number of png outputs
    if output_file.isRGB:
        # Compute combinatorial number of files expected
        shape = [output_file.SizeX, output_file.SizeY, output_file.SizeZ, output_file.SizeC, output_file.SizeT]
        # Remove RGB dimension from combinatorial count
        shape.pop(output_file.RGBDimension)
    else:
        # Compute combinatorial number of files expected
        shape = [output_file.SizeX, output_file.SizeY, output_file.SizeZ, output_file.SizeC, output_file.SizeT]

    total_count = np.prod(shape[2:])
    print shape, output_file.isRGB

    # Make sure same number of files generated
    assert num_png_files == total_count

    # Save to local storage output deepzoom files
    output_file.load_and_save_assets(output_file.file_path, i_xyzct=(-1, -1, -1, -1, -1), n_xyzct=(-1, -1, -1, -1, -1),
                                     asset_format='dzi')

    # Get number of files generated
    num_dzi_files = len(glob.glob('%s-zoomed/*.dzi' % os.path.basename(output_file.file_path)))
    print glob.glob('%s-zoomed/*.dzi' % os.path.basename(output_file.file_path))

    # Ensure correct number of dzi outputs
    if output_file.isRGB:
        # Compute combinatorial number of files expected
        shape = [output_file.SizeX, output_file.SizeY, output_file.SizeZ, output_file.SizeC, output_file.SizeT]
        # Remove RGB dimension from combinatorial count
        shape.pop(output_file.RGBDimension)
    else:
        # Compute combinatorial number of files expected
        shape = [output_file.SizeX, output_file.SizeY, output_file.SizeZ, output_file.SizeC, output_file.SizeT]

    total_count = np.prod(shape[2:])
    print shape

    # Make sure same number of files generated
    assert num_dzi_files == total_count

    # Clean up
    os.system('rm %s-zoomed/*.png' % os.path.basename(output_file.file_path))
    os.system('rm %s-zoomed/*.dzi' % os.path.basename(output_file.file_path))

    mock_s3().stop()
    mock_ssm().stop()
    mock_dynamodb2().stop()
