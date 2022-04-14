#! /bin/bash

if [ -z $1 ]
then
    ! grep -r -P '^\s*# ((?!coding|pylint|flake8))[a-z].*$' fumes/
else
    ! grep -r -P '^\s*# ((?!coding|pylint|flake8))[a-z].*$' $1
fi
