#!/bin/sh

if [ $1 == "1" ]
then
	python3 -W ignore tests.py
	./test
	python3 -W ignore tests_visual.py
	./test_visual
else
	python3 -W ignore tests.py
	./test
fi