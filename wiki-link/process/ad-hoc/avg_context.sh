#!/bin/bash

# run this first: /home/martin/bin/redis-server ./conf/redis.conf

REDIS_HOST=`hostname`
REDIS_PORT=63793

NUM_WORKERS=5

NAME=avg-context

mkdir -p ./json
mkdir -p ./logs/$NAME

# populate Redis.
mvn scala:run -Dlauncher="avg-context-master" > ./logs/$NAME/master.log

# start workers
for i in `seq 1 $NUM_WORKERS`
do
  qsub -cwd -b y -N worker-$i -j y -o ./logs/$NAME /share/apps/maven/bin/mvn scala:run -Dlauncher=avg-context-slave #-DaddArgs=@redis=$REDIS_HOST:$REDIS_PORT
done

