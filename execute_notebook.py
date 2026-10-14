"""
Execute RailWay_AI_Training.ipynb with nbclient to populate genuine cell execution outputs.
"""
import nbformat
from nbclient import NotebookClient

def execute_nb():
    print("Reading RailWay_AI_Training.ipynb...")
    with open("RailWay_AI_Training.ipynb", "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    print("Executing notebook top to bottom...")
    client.execute()

    output_path = "RailWay_AI_Training.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Successfully executed and saved populated notebook to {output_path}.")

if __name__ == "__main__":
    execute_nb()
