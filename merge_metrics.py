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

            if method not in methods:
                methods[method] = {}

            methods[method][name] = {
                'mae': mae,
                'perceptual': perceptual,
                'ssim': ssim,
                'psnr': psnr
            }

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
    methods_names = [method for method in methods.keys() if not method.startswith('ours')]
    common_metrics = {}
    print([set(methods[name].keys()) for name in methods_names])
    print(*[set(methods[name].keys()) for name in methods_names])
    common_names = set.intersection(*[set(methods[name].keys()) for name in methods_names])

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
            only_mean_writer.writerow([method, sum(maes) / len(maes), sum(perceptuals) / len(perceptuals), sum(ssims) / len(ssims), sum(psnrs) / len(psnrs)])

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