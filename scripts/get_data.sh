#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir $DIR/../data
curl -o $DIR/../data/data.tar.gz http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
tar zvfx $DIR/../data/data.tar.gz -C $DIR/../data/
rm $DIR/../data/data.tar.gz
