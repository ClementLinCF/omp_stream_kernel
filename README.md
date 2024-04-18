# omp_stream_kernel

hipcc skt_v0.cpp -o skt0 -fopenmp


hipcc -o sktmpi skt_v2_mpi.cpp -I/opt/ompi/include -pthread -Wl,-rpath -Wl,/opt/ompi/lib -Wl,--enable-new-dtags -L/opt/ompi/lib -lmpi -fopenmp

mpirun --allow-run-as-root -np 8 ./sktmpi
