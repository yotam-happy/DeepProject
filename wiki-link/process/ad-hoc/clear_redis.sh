#!/bin/bash

redis-cli -h compute-1-14 -p 63793 KEYS * | xargs redis-cli -h compute-1-14 -p 63793 DEL
