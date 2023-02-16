echo "mpirun -np 48 ./lab3 380 420 6 0.00025 0.1"
echo "<<<"
mpirun -np 48 ./lab3 380 420 6 0.00025 0.1
echo ">>>"

echo "mpirun -np 6 ./lab3 25 19 700 0.00025 0.1"
echo "<<<"
mpirun -np 6 ./lab3 25 19 700 0.00025 0.1
echo ">>>"

echo "mpirun -np 64 ./lab3 1000 15 15 0.00025 0.1"
echo "<<<"
mpirun -np 64 ./lab3 1000 15 15 0.00025 0.1
echo ">>>"

echo "mpirun -np 35 ./lab3 10 1200 35 0.00025 0.1"
echo "<<<"
mpirun -np 35 ./lab3 10 1200 35 0.00025 0.1
echo ">>>"
