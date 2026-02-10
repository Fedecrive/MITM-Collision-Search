#!/bin/bash
set -u

N=24
NP=4
THREADS=2
ITER=1

echo "Starting stress test N=$N NP=$NP THREADS=$THREADS ITER=$ITER"

for ((i=1; i<=ITER; i++)); do
    log="stress_${i}.log"
    echo -n "Iter $i... "
    
    OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp OMPI_MCA_btl_tcp_if_exclude=lo,docker0 \
    mpirun -np $NP \
    ./mitm --n $N --C0 a9267e3f07e8ac8c --C1 2416912919f36094 > "$log" 2>&1
    
    if grep -q "\[TOTAL\] solutions across all ranks: 0" "$log"; then
        echo "FAIL: 0 solutions found!"
        echo "Log: $log"
        exit 1
    fi
    
    if ! grep -q "\[TOTAL\] solutions across all ranks:" "$log"; then
        echo "FAIL: Crash or missing output!"
        echo "Log: $log"
        exit 1
    fi
    
    if grep -q "DEBUG:" "$log"; then
        echo "DEBUG info found in $log"
        # mv "$log" "debug_${i}.log"
    else
        rm "$log"
    fi
done

echo "Stress test passed."
