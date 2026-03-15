import os
import re
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser()

parser.add_argument(
    '--input', '-i',
    type=str,
    default='1_more.txt',
    help='The Maple output file to be parsed as the input to this transpiler.'
)
parser.add_argument(
    '--output', '-o',
    type=str,
    default='one_more_function',
    help='The name of the output directory or script name to write the torch function(s).'
)

args = parser.parse_args()

input_loc = args.input
output_loc = args.output

solution_pattern = re.compile(r"\[.+?\]")
names = ('E_x', 'E_y', 'E_z', 'B_x', 'B_y', 'B_z')

cond_replacements = OrderedDict({
    'sqrt(x^2+y^2+z^2)': ('r', 'r = torch.sqrt(x**2+y**2+z**2)'),
    'x^2+y^2+z^2': ('r_sq', 'r_sq = x**2+y**2+z**2'),
})

replacements = OrderedDict({
    'sin': 'torch.sin',
    'cos': 'torch.cos',
    'exp': 'torch.exp',
    'sqrt': 'torch.sqrt',
    '^': '**',
})

solution_text = open(args.input).read()

if solution_text.startswith('[['):
    assert solution_text.endswith(']]')
    solution_text = solution_text[1:-1] # strip outer brackets
    if os.path.exists(output_loc) and os.path.isdir(output_loc):
        next_num = min(
            int(x.split('_')[1].split('.')[0])
            for x in os.listdir(output_loc)
        )
    else:
        os.makedirs(output_loc)
    is_list = True
    if output_loc.endswith('.py'):
        output_loc = output_loc[:-3]
else:
    is_list = False
    if not output_loc.endswith('.py'):
        output_loc += '.py'
    
solutions = re.findall(solution_pattern, solution_text)

for i, solution in enumerate(solutions):
    
    if is_list:
        loc = os.path.join(output_loc, f'function_{i+1}.py')
    else:
        loc = output_loc + '.py'
    
    with open(loc, 'w+', encoding='utf-8') as f:
        
        f.write('import torch\nfrom torch import Tensor\n\nc = 1.0\n\n')
        f.write('def u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:\n')
        
        for cond_replacement in cond_replacements:
            if cond_replacement in solution:
                f.write(f'\t{cond_replacements[cond_replacement][1]}\n')
        f.write('\n')
        
        components = solution[1:-1].split(',')
        assert len(components) == 6
        for component, name in zip(components, names):
            for cond_replacement in cond_replacements:
                component = component.replace(cond_replacement, cond_replacements[cond_replacement][0])
            for replacement in replacements:
                component = component.replace(replacement, replacements[replacement])
            if component.strip() == '0':
                component = 'torch.zeros(t.size()).to(t.device)'
            f.write(f'\t{name} = {component}\n')
        
        f.write(f'\n\treturn torch.vstack(({", ".join(name for name in names)}))')
