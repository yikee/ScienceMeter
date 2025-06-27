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


    pres = {"denominator": 0, "succ": 0, "dist": 0, "loss": 0}
    acq  = {"denominator": 0, "succ": 0, "dist": 0, "loss": 0}
    proj = {"denominator": 0, "succ": 0, "loss": 0}

    for pb, pa, nb, na, fb, fa in zip(all_data["prior_before"], all_data["prior_after"],
                                    all_data["new_before"], all_data["new_after"],
                                    all_data["future_before"], all_data["future_after"]):

        # 1) Knowledge Preservation (prior)
        if pb["label"] == "correct" and nb["label"] == "unknown":
            pres["denominator"] += 1
            if pa["label"] == "correct":
                pres["succ"] += 1
            elif pa["label"] == "incorrect":
                pres["dist"] += 1
            else:                         # pa["label"] == "unknown"
                pres["loss"] += 1

        # 2) Knowledge Acquisition (new)
        if nb["label"] == "unknown":
            acq["denominator"] += 1
            if na["label"] == "correct":
                acq["succ"] += 1
            elif na["label"] == "incorrect":
                acq["dist"] += 1
            else:                         # na["label"] == "unknown"
                acq["loss"] += 1

        # 3) Knowledge Projection (future)
        if fb["label"] == "unknown" and nb["label"] == "unknown":
            proj["denominator"] += 1
            if fa["label"] == "correct":
                proj["succ"] += 1
            else:                         # fa["label"] == "unknown"
                proj["loss"] += 1

    safe = lambda x, d: round(x / d, 3) if d else 0.0

    print("-----------------")
    print("Knowledge Preservation:         ", safe(pres["succ"], pres["denominator"]))
    print("distortion in Preservation:     ", safe(pres["dist"], pres["denominator"]))
    print("loss in Preservation:           ", safe(pres["loss"], pres["denominator"]))

    print("-----------------")
    print("Knowledge Acquisition:          ", safe(acq["succ"], acq["denominator"]))
    print("distortion in Acquisition:      ", safe(acq["dist"], acq["denominator"]))
    print("loss in Acquisition:            ", safe(acq["loss"], acq["denominator"]))

    print("-----------------")
    print("Knowledge Projection:           ", safe(proj["succ"], proj["denominator"]))
    print("loss in Projection:             ", safe(proj["loss"], proj["denominator"]))
    print("-----------------")
