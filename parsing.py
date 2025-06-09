import re

def yes_no_parsing(response):
    response = response.lower()
    response = re.sub(r'answer yes or no only', '', response, flags=re.IGNORECASE).strip()

    pattern = r"\$\boxed\{(yes|no)\}\$"
    match = re.search(pattern, response)
    if match:
        return match.group(1)
    
    if response[:2] == "0\n":
        return "no"
    if response[:2] == "1\n":
        return "yes"

    response = re.sub(r'^[^\w]*', '', response)
    if response.startswith("no"):
        return "no"
    if response.startswith("yes"):
        return "yes"
    
    if "the final answer is" in response:
        response = response.split("the final answer is")[-1].strip()
    if "\nresponse:" in response:
        response = response.split("\nresponse:")[-1].strip()

    yes_match = re.search(r'\byes\b', response)
    no_match = re.search(r'\bno\b', response)
    
    if yes_match and not no_match:
        return "yes"
    if no_match and not yes_match:
        return "no"

    # print("fail to parse answer", response, "----------------")
    return "fail to parse"