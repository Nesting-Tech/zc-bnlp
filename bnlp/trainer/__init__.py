# -*- coding: utf-8 -*-
"""
    Created by JoeYip on 29/03/2019

    :copyright: (c) 2019 by nesting.xyz
    :license: BSD, see LICENSE for more details.
"""

import shutil

CHECKPOINT_PATTERN = ".*model.ckpt-([0-9]*)\Z"
TMP_CHECKPOINT_FILE_SUFFIX = "model.tmp.ckpt"


def copy_checkpoint(source, target):
    for ext in (".index", ".data-00000-of-00001"):
        shutil.copyfile(source + ext, target + ext)
