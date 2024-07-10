#~ /bin/bash
set -e
end=$(($1-1))
for (( i=0; i<=$end; i=i+3 ))
do
	qsub run.sh $1 $i;
	qsub run.sh $1 $(($i+1));
	qsub run.sh $1 $(($i+2));
	sleep 20s
done
while [ `ls | grep "point" | wc -l` != "$1" ]
do
	echo "Waiting all the jobs finish..." `ls | grep "point" | wc -l`;
	sleep 25s
done
mv point* run.sh.* ./output
cd ./output
cat point* > points.txt
mv points.txt ../
