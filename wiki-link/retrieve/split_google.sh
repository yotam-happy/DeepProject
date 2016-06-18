#!/bin/bash

echo $1

mkdir google
cd google

for ((i=0; i<10; i+=1)); do cat $1/data-0000$i-of-00010.gz | zcat; done | perl -ne 'if (($s .= $_) =~ s/(.*)\n{3}//s) {$n++; open F, sprintf(">%09d", $n); print F $1; close F}' 

cd ..
