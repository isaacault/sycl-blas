N_min=32 ; N_max=64; N_mlt=2;
N_min=1 ; N_max=1000000000; N_mlt=10;
N_min=1 ; N_max=1048576; N_mlt=2;
N_min=1 ; N_max=67108864; N_mlt=2;
N_min=1 ; N_max=16777216; N_mlt=2;
N_min=1024 ; N_max=8192; N_mlt=2;
N=$N_min
if [ -a resBLAS2.txt ] ; then rm resBLAS2.txt ; fi
while [ $N -le $N_max ] ; do
  echo $N;
  echo $N >> resBLAS2.txt
  ../exec.sh ../build/tests/blas2_clblas_test $N | grep -v "ANALYSIS" > clblas.txt
  ../exec.sh ../build/tests/blas2_interface_test  $N | grep -v "ANALYSIS" | grep -v "considered" | cut -d, -f2- > syclblas.txt
  cat syclblas.txt | cut -d, -f3- > aux.txt
#  ../build/tests/paper_blas1_test  $N > syclblas.txt
#  cat syclblas.txt | cut -d, -f3- > aux.txt
#  paste -d, clblas.txt aux.txt >> resBLAS2.txt
  paste -d, clblas.txt syclblas.txt >> resBLAS2.txt
#  cat clblas.txt syclblas.txt aux.txt
  N=`expr $N \* $N_mlt` 
done

