#!/usr/bin/env python
import glob
import os

import pytest
from base_processor.tests import init_ssm
from base_processor.tests import setup_processor
# our module(s)
from czi_processor import CZIProcessor
from moto import mock_s3
from moto import mock_ssm

test_processor_data = [
    'sample.czi'
]


@pytest.mark.parametrize("filename", test_processor_data)
def test_czi_processor(filename):
    mock_ssm().start()
    mock_s3().start()

    init_ssm()

    # init task
    inputs = {'file': os.path.join('/test-resources', filename), 'optimize_view': "y"}
    task = CZIProcessor(inputs=inputs)

    setup_processor(task)

    # run
    task.run()

    # Check outputs
    first_slice_set = sorted(glob.glob('%s-zoomed/*.dzi' % os.path.basename(task.file)))[0]

    assert os.path.isfile('%s-zoomed/slide.dzi' % os.path.basename(task.file))
    assert os.path.isfile('metadata.json')

    mock_s3().stop()
    mock_ssm().stop()
