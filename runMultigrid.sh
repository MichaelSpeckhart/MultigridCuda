clear

nvcc -rdc=true -o Multigrid MultigridGPU.cu -lcudadevrt 

./Multigrid