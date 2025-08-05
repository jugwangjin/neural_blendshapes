# the files are written with this format

# MAE: 0.04906938147420684
# Perceptual: 0.1904288882513841
# SSIM: 0.8347235894203187
# PSNR: 19.975480047861733
import os
import glob
import csv


# 

def main():
    metrics_dir = './figures/metrics'

    metrics_files = glob.glob(f'{metrics_dir}/*.txt')

    methods = {}

    for metric_file in metrics_files:
        # there are multiple underbars (_)
        # only the first underbar splits the method from the name
        
        # for example, ours_your_name.txt -> method = ours, name = your_name

        method, name = os.path.basename(metric_file).split('_', 1)
        name = name.split('.')[0]
        
        # read the file, collect MAE, Perceptual, SSIM, PSNR

        with open(metric_file, 'r') as f:
            lines = f.readlines()
            mae = float(lines[0].split(': ')[1])
            perceptual = float(lines[1].split(': ')[1])
            ssim = float(lines[2].split(': ')[1])
            psnr = float(lines[3].split(': ')[1])

            # if method == 'oursv13' and name == 'yufeng_metrics':
            #     print(method, name)
            #     continue
            # if method == 'oursv14' and name == 'yufeng_metrics':
            #     print(method, name)
            #     method = 'oursv13'
            #     print(method, name)
            if method not in methods:
                methods[method] = {}


            methods[method][name] = {
                'mae': mae,
                'perceptual': perceptual,
                'ssim': ssim,
                'psnr': psnr
            }

    print(methods.keys())

    # if there are same name in oursv{*} and oursv{*}, pick higher psnr one and save it on oursv13
    
    for method in methods: 
        if method.startswith('oursv') and method != 'oursv13':
            # get the names
            names = methods[method].keys()
            for name in names:
                if name in methods['oursv13']:
                    if methods[method][name]['psnr'] > methods['oursv13'][name]['psnr']:
                        methods['oursv13'][name] = methods[method][name]
                        print(f'oursv13 {name} updated, from {method}')
                else:
                    methods['oursv13'][name] = methods[method][name]
                    print(f'oursv13 {name} added, from {method}')
    to_del = []
    for method in methods:
        if method.startswith('oursv') and method != 'oursv13':
            print(f'{method} removed')
            to_del.append(method)
    for method in to_del:
        del methods[method]

    




    # from method, remove oursv6, ourv10 
    # from name, remove nf_01, nf_03

    methods = {method: metrics for method, metrics in methods.items() if method not in ['oursv6', 'oursv10']}

    # now summarize this into a csv file

    # chunk the metrics based on the methods - split with two blank rows between methods.
    
    with open('figures/metrics/metrics.csv', 'w') as f:
        writer = csv.writer(f)
        # iterate with sorted keys
        for method in sorted(methods.keys()):
            metrics = methods[method]
            writer.writerow([method])
            writer.writerow(['Name', 'MAE', 'Perceptual', 'SSIM', 'PSNR'])
            for name in sorted(metrics.keys()):
                metric = metrics[name]
                writer.writerow([name, metric['mae'], metric['perceptual'], metric['ssim'], metric['psnr']])

            # write mean metrics

            maes = [metric['mae'] for metric in metrics.values()]
            perceptuals = [metric['perceptual'] for metric in metrics.values()]
            ssims = [metric['ssim'] for metric in metrics.values()]
            psnrs = [metric['psnr'] for metric in metrics.values()]

            writer.writerow(['Mean', sum(maes) / len(maes), sum(perceptuals) / len(perceptuals), sum(ssims) / len(ssims), sum(psnrs) / len(psnrs)])

            writer.writerow([])
            writer.writerow([])


    # get the all intersection of metrics over all methods
    # mean of the intersection of metrics over each method 

    # except ours*
    methods_names = [method for method in methods.keys()]
    # methods_names = [method for method in methods.keys()]
    common_metrics = {}
    print([set(methods[name].keys()) for name in methods_names])
    print(*[set(methods[name].keys()) for name in methods_names])
    common_names = set.intersection(*[set(methods[name].keys()) for name in methods_names])

    # remove nf_01 and nf_03 from common_names
    common_names = [name for name in common_names if not name.startswith('nf_')]

    common_names = sorted(list(common_names))
    

    only_mean_csv = open('figures/metrics/metrics_only_mean.csv', 'w')
    only_mean_writer = csv.writer(only_mean_csv)

    only_mean_writer.writerow(['Name', 'MAE', 'Perceptual', 'SSIM', 'PSNR'])

    with open('figures/metrics/metrics_common.csv', 'w') as f:
        writer = csv.writer(f)
        for method in sorted(methods.keys()):
            metrics = methods[method]
            writer.writerow([method])
            writer.writerow(['Name', 'MAE', 'Perceptual', 'SSIM', 'PSNR'])
            for name in common_names:
                if not name in metrics:
                    continue
                writer.writerow([name, metrics[name]['mae'], metrics[name]['perceptual'], metrics[name]['ssim'], metrics[name]['psnr']])
            
            maes = [metrics[name]['mae'] for name in common_names if name in metrics.keys()]
            perceptuals = [metrics[name]['perceptual'] for name in common_names if name in metrics.keys()]
            ssims = [metrics[name]['ssim'] for name in common_names if name in metrics.keys()]
            psnrs = [metrics[name]['psnr'] for name in common_names if name in metrics.keys()]

            writer.writerow(['Mean', sum(maes) / len(maes), sum(perceptuals) / len(perceptuals), sum(ssims) / len(ssims), sum(psnrs) / len(psnrs)])
            only_mean_writer.writerow([
                method,
                f'{sum(maes) / len(maes):.4f}',
                f'{sum(perceptuals) / len(perceptuals):.4f}',
                f'{sum(ssims) / len(ssims):.4f}',
                f'{sum(psnrs) / len(psnrs):.2f}'
            ])

    only_mean_csv.close()

# another version of csv file: for each methods.keys, write the metrics for each name, only PSNR


    with open('figures/metrics/metrics_psnr.csv', 'w') as f:
        writer = csv.writer(f)
        for method in sorted(methods.keys()):
            metrics = methods[method]
            writer.writerow([method])
            writer.writerow(['Name', 'PSNR'])
            for name in sorted(metrics.keys()):
                metric = metrics[name]
                writer.writerow([name, metric['psnr']])

            writer.writerow([])
            writer.writerow([])



        
if __name__ == '__main__':
    main()