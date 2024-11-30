import datasets
from datasets import load_dataset

dataset = load_dataset('clembench-playpen/lmentry',"ends_with_letter")

print(dataset)

