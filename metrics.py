import json
import argparse
import parsing
import random

random.seed(42)

def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-bm", "--basemodel", help='e.g., "llama"')
    argParser.add_argument("-m", "--model", help='e.g., "llama", "it_trainqadoc"')
    argParser.add_argument("-d", "--domain", help='e.g., "computer_science"')
    argParser.add_argument("-t", "--task", help='"judgment" or "generation"')

    args = argParser.parse_args()
    basemodel = args.basemodel
    model = args.model
    domain = args.domain
    task = args.task

    def label_judgment_instance(instance):
        consistency = instance["consistency"]
        linguistic_confidence = True if instance["linguistic_confidence"] == "yes" else False
        moreinfo = False if instance["moreinfo"] == "yes" else True

        model_confidences = [consistency, linguistic_confidence, moreinfo]
        model_confidence = "high" if sum(model_confidences) >= 2 else "low"
        instance["model_confidence"] = model_confidence

        if instance["model_confidence"] == "high" and instance["factual_accuracy"] == "accuract":
            label = "correct"
        elif instance["model_confidence"] == "high" and instance["factual_accuracy"] == "inaccurate":
            label = "incorrect"
        else:
            label = "unknown"
        instance["label"] = label

        return instance

    def label_generation_instance(instance):
        moreinfo = False if instance["moreinfo"] == "yes" else True
        model_confidence = "high" if moreinfo else "low"
        instance["model_confidence"] = model_confidence

        if instance["model_confidence"] == "high" and instance["factual_accuracy"] == "accuract":
            label = "correct"
        elif instance["model_confidence"] == "high" and instance["factual_accuracy"] == "inaccurate":
            label = "incorrect"
        else:
            label = "unknown"
        instance["label"] = label

        return instance
    
    all_data = {}
    for knowledge in ["prior", "new", "future"]:
        before_key = f"{knowledge}_before" # before knowledge updates
        after_key = f"{knowledge}_after" # after knowledge updates
        
        all_data[before_key] = load_json(f"results/{basemodel}/{domain}/{task}/{basemodel}_{knowledge}.json")
        all_data[after_key] = load_json(f"results/{basemodel}/{domain}/{task}/{model}_{knowledge}.json")
        
        assert len(all_data[before_key]) == len(all_data[after_key])
        assert all(b["paperId"] == a["paperId"] for b, a in zip(all_data[before_key], all_data[after_key]))

        if task == "judgment":
            all_data[before_key] = [label_judgment_instance(instance) for instance in all_data[before_key]]
            all_data[after_key] = [label_judgment_instance(instance) for instance in all_data[after_key]]
        elif task == "generation":
            all_data[before_key] = [label_generation_instance(instance) for instance in all_data[before_key]]
            all_data[after_key] = [label_generation_instance(instance) for instance in all_data[after_key]]

    print("-----------------")
    knowledge = "prior"
    before_data, after_data = all_data[f"{knowledge}_before"], all_data[f"{knowledge}_after"]
    denominator = len([b for b in before_data if b["label"] == "correct"])
    success, distortion, loss = 0, 0, 0
    for b, a in zip(before_data, after_data):
        if b["label"] == "correct" and a["label"] == "correct":
            success += 1
        elif b["label"] == "correct" and a["label"] == "incorrect":
            distortion += 1
        elif b["label"] == "correct" and a["label"] == "unknown":
            loss += 1
    print("Knowledge Preservation: ", round(success/denominator, 3))
    print("distortion in Perservation: ", round(distortion/denominator, 3))
    print("loss in Perservation: ", round(loss/denominator, 3))

    print("-----------------")
    knowledge = "new"
    before_data, after_data = all_data[f"{knowledge}_before"], all_data[f"{knowledge}_after"]
    denominator = len([b for b in before_data if b["label"] == "unknown"])
    success, distortion, loss = 0, 0, 0
    for b, a in zip(before_data, after_data):
        if b["label"] == "correct" and a["label"] == "correct":
            success += 1
        elif b["label"] == "correct" and a["label"] == "incorrect":
            distortion += 1
        elif b["label"] == "correct" and a["label"] == "unknown":
            loss += 1
    print("Knowledge Acquisition: ", round(success/denominator, 3))
    print("distortion in Acquisition: ", round(distortion/denominator, 3))
    print("loss in Acquisition:", round(loss/denominator, 3))

    print("-----------------")
    knowledge = "future"
    before_data, after_data = all_data[f"{knowledge}_before"], all_data[f"{knowledge}_after"]
    denominator = len([b for b in before_data if b["label"] == "unknown"])
    success, loss = 0, 0
    for b, a in zip(before_data, after_data):
        if b["label"] == "correct" and a["label"] == "correct":
            success += 1
        elif b["label"] == "correct" and a["label"] == "unknown":
            loss += 1
    print("Knowledge Projection: ", round(success/denominator, 3))
    print("loss in Projection:", round(loss/denominator, 3))
    print("-----------------")