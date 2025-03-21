# PPMLP: A Privacy-Preserving Federated Learning Protocol for Multilayer Perceptrons

We propose a homomorphic-encryption-based privacy-preserving federated learning protocol for multilayer perceptrons. By utilizing neural networks' inherent permutability, we reduce the possibility of client collusion attacks. The protocol allows multiple distributed clients to collaboratively train a machine learning model without sharing their private data, addressing ongoing concerns regarding data privacy during the federated learning process. 

## Dependencies

keras == "2.14.0"

tensorflow == "2.14.1"

matplotlib == "3.6.2"

openfhe == "1.2.0"

openfhe-python == "0.8.8"

## Usage

We use the MNIST dataset to test our protocol. You can optionally partition the dataset using three different methods: IID (PARTITION NO. = 0), Non-IID (PARTITION NO. = 1), Non-IID-Dirichlet (PARTITION NO. = 2). Run the code using the following command:

    python3 main.py PARTITION NO.

The script, by default, parititons the data amongst 3 clients and runs a total of 100 federated learning rounds. 

