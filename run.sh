

make all
rm -f matrix_*.txt

echo ""
echo "<<<"
mpirun -np 2 ./lab3 9 9 360 0.00025 0.1
echo ">>>"

echo ""
echo "<<<"
mpirun -np 3 ./lab3 5 13 300 0.01 1
echo ">>>"

echo ""
echo "<<<"
mpirun -np 3 ./lab3 13 5 300 0.01 1
echo ">>>"

make clean