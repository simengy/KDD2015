#! /bin/bash

sed  -i 's/nrows=[0-9]\{1,\})/nrows=None)/' read*.py
cat feature_engineering.sh | parallel -j6 {}

chmod 440 feature/*.csv
