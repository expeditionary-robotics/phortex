#! /bin/bash

if [ -z $1 ]
then
    echo "Running pylint..."
    pylint fumes
    echo "Running flake8..."
    flake8 fumes
    echo "Running pydocstyle..."
    pydocstyle fumes
    echo "Running comment capitalization checker..."
    ./lint_comments.sh fumes
else
    echo "Running pylint..."
    pylint $1
    echo "Running flake8..."
    flake8 $1
    echo "Running pydocstyle..."
    pydocstyle $1
    echo "Running comment capitalization checker..."
    ./lint_comments.sh $1
fi
