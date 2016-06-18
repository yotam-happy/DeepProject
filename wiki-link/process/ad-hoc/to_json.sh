#!/bin/bash

# run this first: /home/martin/bin/redis-server ./conf/redis.conf

REDIS_HOST=`hostname`
REDIS_PORT=63793

NAME=full-superset

JSON_DIR=./json-$NAME
LOGS_DIR=./logs/to_json-$NAME

NUM_WORKERS=100

mkdir -p $JSON_DIR
mkdir -p $LOGS_DIR

# populate Redis with the google data.
mvn scala:run -Dlauncher="write-json-redis-master" -DaddArgs=@json=$JSON_DIR > $LOGS_DIR/master.log

# start workers
for i in `seq 1 $NUM_WORKERS`
do
  qsub -cwd -b y -N worker-$i -j y -o $LOGS_DIR /share/apps/maven/bin/mvn scala:run -Dlauncher=write-json-redis-slave -DaddArgs=@json=`pwd`/$JSON_DIR #\|@redis=$REDIS_HOST:$REDIS_PORT
done

