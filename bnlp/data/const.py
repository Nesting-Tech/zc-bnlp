# -*- coding: utf-8 -*-
"""
    Created by JoeYip on 29/03/2019

    :copyright: (c) 2019 by nesting.xyz
    :license: BSD, see LICENSE for more details.
"""

from enum import Enum, unique


@unique
class Language(Enum):
    en = "english"
    zh = "chinese"
    ar = "arabic"


@unique
class DataSetType(Enum):
    train = 'train'
    dev = 'dev'
    test = 'test'
