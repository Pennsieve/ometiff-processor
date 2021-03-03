#!/usr/bin/env python
import os
import json

import pytest
from moto import mock_s3
from moto import mock_ssm
# our module(s)
from ometiff_processor import OMETIFFProcessor, OMETIFFOutputFile

from base_processor.tests import init_ssm
from base_processor.tests import setup_processor

test_processor_data = [
    '4D-series.ome.tiff',
    'color.tif',
    # 'movie.tif',
    'multi-channel.ome.tiff',
    'multi-channel-4D-series.ome.tiff',
    'multi-channel-time-series.ome.tiff',
    'multi-channel-z-series.ome.tiff',
    'single-channel.ome.tiff',
    'time-series.ome.tiff',
    'z-series.ome.tiff',
    # Large page example
    'multipage_tif_example.tif'
]


@pytest.mark.parametrize("filename", test_processor_data)
def test_ometiff_processor(filename):
    mock_ssm().start()
    mock_s3().start()

    init_ssm()

    # init task
    inputs = {'file': os.path.join('/test-resources', filename)}
    task = OMETIFFProcessor(inputs=inputs)

    setup_processor(task)

    # run
    task.run()

    # Check outputs
    assert os.path.isfile('view_asset_info.json')
    json_dict = json.load(open('view_asset_info.json'))
    assert 'size' in json_dict.keys()
    assert 'fileType' in json_dict.keys()

    assert os.path.isdir('%s-zoomed' % os.path.basename(task.outputs[0].file_path))
    assert os.path.isfile('%s-zoomed/dimensions.json' % os.path.basename(task.outputs[0].file_path))
    assert os.path.isfile('metadata.json')

    # Check hack for backwards compatibility with viewer
    assert os.path.isfile('%s-zoomed/slide.dzi' % os.path.basename(task.outputs[0].file_path))
    # assert os.path.isfile('%s-zoomed/slide.png' % os.path.basename(task.outputs[0].file_path))

    mock_s3().stop()
    mock_ssm().stop()
