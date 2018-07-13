#!/usr/bin/env bash

export SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $SCRIPT_DIR

aws s3 sync results s3://gan-generated-mnist/

sudo shutdown -h now
