#!/usr/bin/env bash

export SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $SCRIPT_DIR

sudo python3 -m gan.train --epochs $1
aws s3 sync bin s3://mnist-gan-binaries/

sudo shutdown -h now
