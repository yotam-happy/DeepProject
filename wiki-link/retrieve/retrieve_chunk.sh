#!/bin/bash

LOG_DIR=logs/`basename $1`
OUTPUT_DIR=pages/`basename $1`

echo $LOG_DIR
echo $OUTPUT_DIR

mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

if [ ! -e "$LOG_DIR/complete" ]
then
        if [ ! -e "$LOG_DIR/working" ]
        then
                #remove potentially, partially downloaded chunk
                rm -r $LOG_DIR
                rm -r $OUTPUT_DIR
                mkdir -p $LOG_DIR
                mkdir -p $OUTPUT_DIR

                touch $LOG_DIR/working

                while read line
                do
                  ID=`echo $line | awk '{ print $1 }'`
                  URL=`echo $line | awk '{ print $2 }'`
                  echo $URL > $LOG_DIR/$ID.url
                  wget --timeout=20 --tries=2 -o $LOG_DIR/$ID -O $OUTPUT_DIR/$ID $URL
                done < "$1"

                touch $LOG_DIR/complete
                rm $LOG_DIR/working
        else
                echo "Skipping in-progress chunk:  " `basename $1`
        fi
else
        echo "Skipping complete chunk:  " `basename $1`
fi
