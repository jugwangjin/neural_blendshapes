import os



if __name__ == '__main__':

    dire = '/Bean/log/gwangjin/2024/nbshapes_comparisons/ours_enc_v10'

    for subd in os.listdir(dire):
        # if subd == 'malte_1':
            # continue/
        # if subd == 'obama':
            # continue
        # if subd == 'subject_3':
        #     continue

        if os.path.exists(os.path.join(dire, subd, 'final_eval.txt')):
            continue
            
        if os.path.exists(os.path.join(dire, subd, 'stage_1', 'images', 'grid', 'grid_1000_expression.png')):
            print(subd)
            with open(os.path.join(dire, subd, 'currently_training.txt'), 'w') as f:
                f.write('')

    # input_dir = /Bean/data/gwangjin/2024/nbshapes/flare/yufeng/yufeng
    # train_dir = ["MVI_1814", "