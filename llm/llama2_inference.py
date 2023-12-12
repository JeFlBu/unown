from llama_cpp import Llama
from datasets import load_from_disk
import copy
from tqdm import tqdm
import pandas as pd
import pickle


"""
---------------------------------------------------
UTILS FUNCTIONS
---------------------------------------------------
"""

def positive_template(sample):
    return f"""
Write a claim that uses the following evidence.

Evidence:
<title> {sample["title"]} <evidence> {".".join(sample["evidences_txt"]).strip()}

Claim:
""".strip()

def negative_template(sample):
    return f"""
Write a negative claim that is false with regards to the following evidence.

Evidence:
<title> {sample["title"]} <evidence> {".".join(sample["evidences_txt"]).strip()}

Claim:
""".strip()

def get_evidence_id(dataset):
    global_ids = []
    for sample in tqdm(dataset):
        sample_sentences = [x for x in sample["evidences_txt"] if x is not None]
        ids = []
        for ev in sample_sentences:
            if ev is not None:
                for s in sample["order"]:
                    if s.startswith("sent"):
                        if sample[s] == ev:
                            ids.append(s.partition("_")[-1])
                            break
        assert len(ids) == len(sample_sentences)
        global_ids.append(ids)
    return global_ids

def generate_claim(llm, prompt):
    return llm(prompt, max_tokens=1024, temperature=0.15, top_p=0.01, top_k=40, repeat_penalty=1.1, stop=["\n\n"])["choices"][0]["text"]

# ---------------------------------------------------

print("Positive claim generation...")

pos_inference_data = load_from_disk("./datasets/inference/raw_dataset_random_pos")["train"]

pos_llm = Llama(
    model_path="./merged_models/pos/positive-claim-generator-feverous-7b-f16-q4_1.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,
)

# Check if pos_ids.pkl exists, if yes load ids from file, if not compute them
pos_ids_path = "./pos_ids.pkl"
try:
    with open(pos_ids_path, "rb") as f:
        ids = pickle.load(f)
except FileNotFoundError:
    ids = get_evidence_id(pos_inference_data)
    with open(pos_ids_path, "wb") as f:
        pickle.dump(ids, f)

preds = []
for sample in tqdm(pos_inference_data):
    # Assert that the title and evidences_txt keys exist
    assert "title" in sample.keys()
    assert "evidences_txt" in sample.keys()
    # Remove from evidences_txt all the None items
    sample["evidences_txt"] = [x for x in sample["evidences_txt"] if x is not None]
    sample_prompt = positive_template(sample)
    try:
        pred = generate_claim(pos_llm, sample_prompt)
        preds.append(pred)
    except TypeError as e:
        print(e)
        continue

# Create a dataframe zipping the ids and the predictions
df = pd.DataFrame(zip(ids, preds), columns=["id", "prediction"])
# Remove the rows where the prediction is None
df = df[df["prediction"].notnull()]
# Save the dataframe to a pickle
df.to_pickle("./positive_predictions.pkl")
# Print the number or rows
print(len(df))

# ---------------------------------------------------

print("Negative claim generation...")

neg_inference_data = load_from_disk("./datasets/inference/raw_dataset_random_neg")["train"]

neg_llm = Llama(
    model_path="./merged_models/neg/negative-claim-generator-feverous-7b-f16-q4_1.gguf",
    n_ctx=1024,
    n_gpu_layers=-1,
)

# Check if neg_ids.pkl exists, if yes load ids from file, if not compute them
neg_ids_path = "./neg_ids.pkl"
try:
    with open(neg_ids_path, "rb") as f:
        ids = pickle.load(f)
except FileNotFoundError:
    ids = get_evidence_id(neg_inference_data)
    with open(neg_ids_path, "wb") as f:
        pickle.dump(ids, f)

preds = []
for sample in tqdm(neg_inference_data):
    # Assert that the title and evidences_txt keys exist
    assert "title" in sample.keys()
    assert "evidences_txt" in sample.keys()
    # Remove from evidences_txt all the None items
    sample["evidences_txt"] = [x for x in sample["evidences_txt"] if x is not None]
    sample_prompt = negative_template(sample)
    try:
        pred = generate_claim(neg_llm, sample_prompt)
        preds.append(pred)
    except TypeError as e:
        continue

# Create a dataframe zipping the ids and the predictions
df = pd.DataFrame(zip(ids, preds), columns=["id", "prediction"])
# Save the dataframe to a pickle
df.to_pickle("./negative_predictions.pkl")
# Print the number or rows
print(len(df))