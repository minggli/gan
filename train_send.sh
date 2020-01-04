#!/usr/bin/env bash

export SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $SCRIPT_DIR

sudo python3 -m gan.train --epochs $1
gsutil -m cp -r ./bin gs://gan-savedmodels

sudo shutdown -h now
