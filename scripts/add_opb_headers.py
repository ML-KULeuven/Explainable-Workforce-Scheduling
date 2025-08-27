
HEADER = \
"""*
* OPB encoding of Workforce Allocation problem, objective is to minimize the number of teams used.
* Online repository:
*   https://github.com/ML-KULeuven/Explainable-Workforce-Scheduling
*
* Original instance: {instance}.json
* Static symmetry breaking: {symmbreak}
* 
* Generated using CPMpy with Exact interface.
*
*
"""

import os

if __name__ == "__main__":

    dirname = "../data/opb_instances_symmbreak"
        
    for fname in os.listdir(dirname):
        if not fname.endswith(".opb"):
            continue

        instance_name = fname.split(".")[0]
        symmbreak = "symmbreak" in dirname
        
        # add header to the file
        with open(os.path.join(dirname, fname), "r") as f:
            content = f.read()
        with open(os.path.join(dirname, fname), "w") as f:
            lines = content.splitlines()
            lines = [lines[0]] + HEADER.format(instance=instance_name, symmbreak=symmbreak).splitlines() + lines[1:]
            f.write("\n".join(lines))

                