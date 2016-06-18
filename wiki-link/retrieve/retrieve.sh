#!/bin/bash

DATA_FILE=""
URLS_FILE=wiki-link-urls.dat
SPLIT_DIR=split_urls

if [ $# -ne 1 ]
then
	echo "Usage:  ./`basename $0`   wiki-link-data-file"
	exit 1
else
	DATA_FILE=$1
fi

# filter the data file URL lines only (should combine sed into one)
if [ -e "$URLS_FILE" ]
then
	echo "$URLS_FILE already exists... skipping creation"
else
	echo "Filtering URL lines to $URLS_FILE..."
	cat $DATA_FILE | sed '/^URL\t/!d' | sed 's/URL\t//' | nl > $URLS_FILE
fi

# split the URLs
if [ -e "$SPLIT_DIR" ]
then
	echo "URLs already split... skipping"
else
	echo "Splitting URLS..."
	mkdir $SPLIT_DIR
	cd $SPLIT_DIR
	split --numeric-suffixes --suffix-length=5 --lines=1000 ../$URLS_FILE 0
	cd ..
fi

echo "Checking output directories..."
mkdir -p logs
mkdir -p pages

# run on each chunk of urls
echo "Running 1 chunk at a time..."
trap "kill 0" SIGINT SIGTERM EXIT
find ./$SPLIT_DIR/ -name "*" -type f | xargs -n 1 --max-procs=150 ./retrieve_chunk.sh
