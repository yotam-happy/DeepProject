#!/bin/bash

# run this first: /home/martin/bin/redis-server ./conf/redis.conf

REDIS_HOST=`hostname`
REDIS_PORT=63793

NUM_WORKERS=150

mkdir -p ./json
mkdir -p ./logs/to_json

# populate Redis.
#mvn scala:run -Dlauncher="write-json-redis-master" > ./logs/to_json/master.log

# start workers
for i in `seq 50 $NUM_WORKERS`
do
  qsub -cwd -b y -N worker-$i -j y -o ./logs/ /share/apps/maven/bin/mvn scala:run -Dlauncher=write-json-redis-slave #-DaddArgs=@redis=$REDIS_HOST:$REDIS_PORT
done

