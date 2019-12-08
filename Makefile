all:
	g++ -fopenmp MLP_Network.cpp  MLP_Layer.cpp MNIST.cpp main.cpp -o mlp
clean:
	rm -f mlp
