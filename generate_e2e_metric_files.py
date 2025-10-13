import torch

path = './data-data/tmp/e2e-rst.pt'
data = torch.load(path)
data.keys()

prefix_unique = list(set(data['prefix']))

rst = []
for prefix in prefix_unique:
    example_rst = {}
    example_rst['prefix'] = prefix
    example_rst['pred'] = []
    example_rst['ref'] = []
    for i in range(len(data['prefix'])):
        if data['prefix'][i] == prefix:
            example_rst['pred'].append(data['pred'][i])
            example_rst['ref'].append(data['ref'][i])

    example_rst['pred'] = list(set(example_rst['pred']))
    rst.append(example_rst)

ref_file_name = './data-data/tmp/llama-e2e-ref-file.txt'
pred_file_name = './data-data/tmp/llama-e2e-pred-file.txt'

ref_file_str = ""
pred_file_str = ""

for i, example in enumerate(rst):
    pred_file_str += example['pred'][0].replace('\n', ' ') + '\n'
    for ref in example['ref']:
        ref_file_str += ref + '\n'
    ref_file_str += '\n'

with open(ref_file_name, 'w') as file:
    file.write(ref_file_str)

with open(pred_file_name, 'w') as file:
    file.write(pred_file_str)


n = 1111

print(data['prefix'][n])
print()
print(data['ref'][n])
print()
print(data['pred'][n])