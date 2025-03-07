

names = ['justin', 'malte_1', 'biden', 'nha_person_0000', 'nha_person_0004', 'wojtek_1', 'yufeng', 'subject_3', 'marcel' 'bala']

# numbers_for_each_name

names_numbers = {}
for i, name in enumerate(names):
    names_numbers[name] = i

numbers = {}

numbers['gt_0.png'] = 0
numbers['shading_no_personalization.png'] = 1
numbers['no_personalization_rgb_0.png'] = 2
numbers['rgb_0.png'] = 4
numbers['shading.png'] = 5


import os

for file in os.listdir('./figures/tracker_effect_selected'):
    os.makedirs('./figures/tracker_effect_selected_2', exist_ok=True)

    if file.startswith('tracker'):
        os.system(f'cp ./figures/tracker_effect_selected/{file} ./figures/tracker_effect_selected_2/{file}')
        continue

    if not file.endswith('.png'):
        continue

    for name in names:
        if file.startswith(name):
            number1 = names_numbers[name]
            break

    if file.endswith('gt_0.png'):
        number2 = 0

    elif file.endswith('shading_no_personalization.png'):
        number2 = 1
    
    elif file.endswith('no_personalization_rgb_0.png'):
        number2 = 2

    elif file.endswith('rgb_0.png'):
        number2 = 4

    elif file.endswith('shading.png'):
        number2 = 3

    os.system(f'cp ./figures/tracker_effect_selected/{file} ./figures/tracker_effect_selected_2/tracker_personalization_{number1}_{number2}.png')