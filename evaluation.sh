# 63 % (par ligne tester par colonne)
# echo "mpirun -np 48 ./lab3 380 420 6 0.00025 0.1"
# echo "<<<"
# mpirun -np 48 ./lab3 380 420 6 0.00025 0.1
# echo ">>>"
# 71%
# echo "mpirun -np 6 ./lab3 25 19 700 0.00025 0.1"
# echo "<<<"
# mpirun -np 6 ./lab3 25 19 700 0.00025 0.1
# echo ">>>"
# 20 % comprend pas pk
# echo "mpirun -np 64 ./lab3 1000 15 15 0.00025 0.1"
# echo "<<<"
# mpirun -np 64 ./lab3 1000 15 15 0.00025 0.1
# echo ">>>"
# TBD apres l'implementation col major
echo "mpirun -np 35 ./lab3 10 1200 35 0.00025 0.1"
echo "<<<"
mpirun -np 35 ./lab3 10 1200 35 0.00025 0.1
echo ">>>"
