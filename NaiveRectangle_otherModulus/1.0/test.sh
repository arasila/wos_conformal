#! /bin/bash

X=2
Y=2
Z=2
for X in {2..50};
do 
  for Y in {2..50};
  do 
    for Z in {2..50};
    do
        gcc -std=c99 -Wall -Werror -pedantic-errors -lm \
	    -DTEST -DBATCH -DX_SLIDES=$X -DY_SLIDES=$Y -DZ_SLIDES=$Z source.c && ./a.out;    
    done
  done
done 
