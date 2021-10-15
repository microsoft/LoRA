import os
import sys
import json
import torch
from collections import OrderedDict

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    model = torch.load(input_path)

    lora = OrderedDict()
    for name, value in model.items():
        if 'lora' in name or (not name.startswith('deberta') and not name.startswith('roberta')):
            new_name = name.replace('self.query_lora_a.weight', 'self.query_proj.lora_A').replace('self.query_lora_b.weight', 'self.query_proj.lora_B').replace('self.value_lora_a.weight', 'self.value_proj.lora_A').replace('self.value_lora_b.weight', 'self.value_proj.lora_B')
            lora[new_name] = value
    print("Save:")
    print(lora.keys())
    folder = os.path.dirname(output_path)
    if not os.path.exists(folder):
        os.mkdir(folder)
    torch.save(lora, output_path)
