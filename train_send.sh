#!/usr/bin/env bash

export SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $SCRIPT_DIR

python -m gan.train
aws s3 sync bin s3://mnist-gan-binaries/

sudo shutdown -h now
