from typing import Literal

def extract(
    input_dir: str = "input",
    output_dir: str = "output",
    backend: Literal["local", "modal"] = "local",
    debug: bool = False
):
    match backend: 
        case "local":
            print("Running plextract locally...")
        case "modal":
            print("Running plextract remotely on modal...")
            
        case _:
            raise Exception(f'Unknown option {backend}. The only valid options are: "local", "modal"')
            
    