N_min=1 ; N_max=1000000000; N_mlt=10;
N_min=1 ; N_max=1048576; N_mlt=2;
N_min=1 ; N_max=67108864; N_mlt=2;
N_min=32 ; N_max=64; N_mlt=2;

N_min=1 ; N_max=16777216; N_mlt=2;
N=$N_min
if [ -a resCG.txt ] ; then rm resCG.txt ; fi
while [ $N -le $N_max ] ; do
  echo $N;
#  echo $N >> res.txt 
  ../exec.sh ../build/tests/cg_test $N > cg.txt
  cat cg.txt
  resCG="$N "
  for time in `tail -n5 cg.txt | cut -d= -f4` ; do
    resCG="$resCG, $time"
  done
  echo $resCG >> resCG.txt
#  cat syclblas.txt | cut -d, -f3- > aux.txt
#  ../build/tests/paper_blas1_test  $N > syclblas.txt
#  cat syclblas.txt | cut -d, -f3- > aux.txt
#  paste -d, clblas.txt aux.txt >> res.txt
#  paste -d, clblas.txt syclblas.txt >> res.txt
#  cat clblas.txt syclblas.txt aux.txt
  N=`expr $N \* $N_mlt` 
done

