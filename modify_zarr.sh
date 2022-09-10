#!/bin/bash
# Modify your zarr

bnlzarr=${HOME}/miniconda3/envs/pymb22a/lib/python3.9/site-packages/zarr

myzarr=$bnlzarr

d=$(date +%Y-%m-%d)
echo "$d"
cp ${myzarr}/core.py ${myzarr}/core-${d}.py
cp core.py ${myzarr}/core.py


