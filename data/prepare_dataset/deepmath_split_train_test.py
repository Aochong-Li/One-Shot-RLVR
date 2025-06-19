from datasets import load_dataset

def split_deepmath_dataset(test_ratio = 0.2, new_dataset_name = "aochongoliverli/DeepMath-103K-split"):
    dataset = load_dataset("zwhe99/DeepMath-103K")["train"]
    dataset = dataset.train_test_split(test_ratio)
    dataset = dataset.remove_columns(["r1_solution_1", "r1_solution_2", "r1_solution_3"])
    dataset.push_to_hub(new_dataset_name)
    
if __name__ == "__main__":
    split_deepmath_dataset(test_ratio = 0.2, new_dataset_name = "aochongoliverli/DeepMath-103K-split")

