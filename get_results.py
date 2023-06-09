from metric.vrr_eval import get_many_medium_few_scores


if __name__ == '__main__':
    dataset_name = 'gvqa'
    csv_file = './test/test.csv'
    cutoffs = [0.8, 0.95]
    if dataset_name == 'vg8k':
        data_dir = 'data/vg8k/'
        ann_dir = 'data/vg8k/seed3/'
    elif dataset_name == 'gvqa':
        data_dir = 'data/gvqa/'
        ann_dir = 'data/gvqa/seed0/'
    
    get_many_medium_few_scores(csv_file, cutoffs, dataset_name, data_dir, ann_dir, syn=True)