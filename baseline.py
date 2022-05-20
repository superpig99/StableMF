import argparse
import os
from surprise import NMF, SVD 
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import PredefinedKFold

parser = argparse.ArgumentParser(description='Baseline')
parser.add_argument ('--dataset', type=str, default='ml-100k', help = 'ml-100k or DoubanMusic')
parser.add_argument('--datapath', type=str, default='./datasets/ml-100k/', help='path to dataset')
parser.add_argument('--arch', default='FunkSVD', choices=['NMF', 'FunkSVD', 'Biased'], help='baseline (default: SVD)')
parser.add_argument('--iter', type=int, default=50, help='number of iterations')
parser.add_argument('--measure', type=list, default=['mae' , 'mse'], help='performance measures')
parser.add_argument('--verbose', type=bool, default=False, help='whether print computed value')


args = parser.parse_args()

files_dir = os.path.expanduser(args.datapath)
train_file = files_dir + 'u%d.base'
test_file = files_dir + 'u%d.test'
folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]    # 5 cross-validate

if args.dataset == 'ml-100k':   
    reader = Reader('ml-100k')
elif args.dataset == 'DoubanMusic':
    reader = Reader(line_format='user item rating', sep='\t')
    
data = Dataset.load_from_folds(folds_files, reader=reader)
pkf = PredefinedKFold()

if args.arch == 'NMF':
    algo = NMF()
elif args.arch == 'FunkSVD':
    algo = SVD(biased=False)
elif args.arch == 'BiasedSVD':
    algo = SVD(biased=True)

errs = dict()  
for measure in args.measure:
    errs[measure] = 0
    
for trainset, testset in pkf.split(data):
    for i in range(0, args.iter):
        algo.fit(trainset)
        predictions = algo.test(testset)
        for measure in args.measure:
            if measure.lower() == 'mae':
                errs[measure] = errs[measure] + accuracy.mae(predictions, verbose=args.verbose)
            elif measure.lower() == 'mse':
                errs[measure] = errs[measure] + accuracy.mse(predictions, verbose=args.verbose)
            elif measure.lower() == 'rmse':
                errs[measure] = errs[measure] + accuracy.rmse(predictions, verbose=args.verbose)
for measure in args.measure:             
    print('{} of {} iterations for 5 cross-validate:{}'.format(measure, args.iter, errs[measure] / (args.iter * 5)))