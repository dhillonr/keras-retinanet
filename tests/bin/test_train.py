"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras_retinanet.bin.train
import keras.backend

import warnings

import pytest

from attrdict import AttrMap


@pytest.fixture(autouse=True)
def clear_session():
    # run before test (do nothing)
    yield
    # run after test, clear keras session
    keras.backend.clear_session()


def test_coco():
    # ignore warnings in this test
    warnings.simplefilter('ignore')

    # args 
    args = AttrMap({
        'epochs' : 1,
        'steps' : 1,
        'imagenet_weights': False,
        'snapshots' : False,
        'dataset_type' : 'coco',
        'coco_path': 'tests/test-data/coco',
    })
    # run training / evaluation
    keras_retinanet.bin.train.main_(args)


def test_pascal():
    # ignore warnings in this test
    warnings.simplefilter('ignore')

    # args 
    args = AttrMap({
        'epochs' : 1,
        'steps' : 1,
        'imagenet_weights': False,
        'snapshots' : False,
        'dataset_type' : 'pascal',
        'pascal_path': 'tests/test-data/pascal'
    })

    # run training / evaluation
    keras_retinanet.bin.train.main_(args)


def test_csv():
    # ignore warnings in this test
    warnings.simplefilter('ignore')

    args = AttrMap({
        'epochs' : 1,
        'steps' : 1,
        'imagenet_weights': False,
        'snapshots' : False,
        'dataset_type' : 'csv',
        'annotations' : 'tests/test-data/csv/annotations.csv',
        'classes': 'tests/test-data/csv/classes.csv'
    })

    # run training / evaluation
    keras_retinanet.bin.train.main_(args)


def test_vgg():
    # ignore warnings in this test
    warnings.simplefilter('ignore')

    args = AttrMap({
        'backbone' : 'vgg16',
        'epochs' : 1,
        'steps' : 1,
        'imagenet_weights': False,
        'snapshots' : False,
        'freeze_backbone': True,
        'dataset_type' : 'coco',
        'coco_path': 'tests/test-data/coco'
    })

    # run training / evaluation
    keras_retinanet.bin.train.main_(args)
