import csv
import json
import pandas as pd
import numpy as np


def generate_csv_file_from_det_obj(detections, csv_path, obj_categores, prd_categories, obj_freq_dict, prd_freq_dict):
    with open(csv_path, 'w', newline='') as csvfile:
        total_test_iters = len(detections)
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        spamwriter.writerow(['image_id',
                             'gt_rel',
                             'det_rel',
                             'rel_freq_gt',
                             'rel_freq_det',
                             'rel_rank',
                             'gt_sbj',
                             'det_sbj',
                             'sbj_freq_gt',
                             'sbj_freq_det',
                             'sbj_rank',
                             'gt_obj',
                             'det_obj',
                             'obj_freq_gt',
                             'obj_freq_det',
                             'obj_rank'])

        for i in range(0, total_test_iters):
            image_id = detections[i]['image'].split('.')[0]
            
            det_scores_prd_all = detections[i]['prd_scores_out'][:, 1:]
            det_scores_sbj_all = detections[i]['sbj_scores_out']
            det_scores_obj_all = detections[i]['obj_scores_out']
            
            det_labels_rel_all = np.argsort(-det_scores_prd_all, axis=1)
            det_labels_sbj_all = np.argsort(-det_scores_sbj_all, axis=1)
            det_labels_obj_all = np.argsort(-det_scores_obj_all, axis=1)

            for j in range(len(detections[i]['gt_sbj_labels'])):
                gt_labels_sbj_idx = detections[i]['gt_sbj_labels'][j]
                gt_labels_obj_idx = detections[i]['gt_obj_labels'][j]
                gt_labels_rel_idx = detections[i]['gt_prd_labels'][j]

                gt_labels_sbj = obj_categores[detections[i]['gt_sbj_labels'][j]]
                gt_labels_obj = obj_categores[detections[i]['gt_obj_labels'][j]]
                gt_labels_rel = prd_categories[detections[i]['gt_prd_labels'][j]]

                det_labels_sbj = obj_categores[det_labels_sbj_all[j, 0]]
                sbj_rank = np.where(det_labels_sbj_all[j, :] == gt_labels_sbj_idx)[0][0]

                det_labels_obj = obj_categores[det_labels_obj_all[j, 0]]
                obj_rank = np.where(det_labels_obj_all[j, :] == gt_labels_obj_idx)[0][0]

                det_labels_rel = prd_categories[det_labels_rel_all[j, 0]]
                rel_rank = np.where(det_labels_rel_all[j, :] == gt_labels_rel_idx)[0][0]

                if gt_labels_sbj in obj_freq_dict.keys():
                    gt_sbj_freq = obj_freq_dict[gt_labels_sbj]
                else:
                    gt_sbj_freq = 0
                if gt_labels_obj in obj_freq_dict.keys():
                    gt_obj_freq = obj_freq_dict[gt_labels_obj]
                else:
                    gt_obj_freq = 0
                if gt_labels_rel in prd_freq_dict.keys():
                    gt_rel_freq = prd_freq_dict[gt_labels_rel]
                else:
                    gt_rel_freq = 0

                if det_labels_sbj in obj_freq_dict.keys():
                    det_sbj_freq = obj_freq_dict[det_labels_sbj]
                else:
                    det_sbj_freq = 0
                if det_labels_obj in obj_freq_dict.keys():
                    det_obj_freq = obj_freq_dict[det_labels_obj]
                else:
                    det_obj_freq = 0
                if det_labels_rel in prd_freq_dict.keys():
                    det_rel_freq = prd_freq_dict[det_labels_rel]
                else:
                    det_obj_freq = 0

                spamwriter.writerow(
                    [image_id,

                     gt_labels_rel,
                     det_labels_rel,
                     gt_rel_freq,
                     det_rel_freq,
                     rel_rank,

                     gt_labels_sbj,
                     det_labels_sbj,
                     gt_sbj_freq,
                     det_sbj_freq,
                     sbj_rank,

                     gt_labels_obj,
                     det_labels_obj,
                     gt_obj_freq,
                     det_obj_freq,
                     obj_rank])
                
                
def get_metrics_from_csv(csv_file, get_mr=False):
    verbose = True
    collected_simple_means = dict()
    collected_per_class_means = dict()
    print('Reading csv file')
    df = pd.read_csv(csv_file)
    print('Done')
    metric_type = 'top1'
    all_prediction_types = ['rel', 'obj', 'sbj']
    gt_prefix = 'gt'
    
    for prediction_type in all_prediction_types:
        df[prediction_type + '_' + metric_type] = df[prediction_type + '_rank'] < int(metric_type[3:])

    df['triplet_top1'] = df['rel_top1'] & df['sbj_top1'] & df['obj_top1']

    if verbose:
        print('------', metric_type, '------')

    # Overall Accuracy
    for prediction_type in all_prediction_types:
        mu = (len(df[df[prediction_type + '_rank'] < int(metric_type[3:])]) / len(df)) * 100.0

        if verbose:
            print('simple-average', prediction_type, '{:2.2f}'.format(mu))

        # collected_simple_means[(csv_file, prediction_type, metric_type)] = mu
        collected_simple_means[f'{csv_file}_{prediction_type}_{metric_type}'] = mu
    print()
    if get_mr:
        # Overall Mean Rank
        for prediction_type in all_prediction_types:
            mu = df[prediction_type + '_rank'].mean() * 100.0 / 250.0
  
            if verbose:
                print('overall-mr', prediction_type, '{:2.2f}'.format(mu))
           
    print()
    
    # Per-class Accuracy
    for prediction_type in all_prediction_types:
        mu = df.groupby(gt_prefix + '_' + prediction_type)[prediction_type + '_' + metric_type].mean().mean() * 100
        if verbose:
            print('per-class-average', prediction_type, '{:2.2f}'.format(mu))
        # collected_per_class_means[(csv_file, prediction_type, metric_type)] = mu
        collected_per_class_means[f'{csv_file}_{prediction_type}_{metric_type}'] = mu
        
    print()
    
    if get_mr:
        # Per-class Mean Rank
        for prediction_type in all_prediction_types:
            mu = df.groupby(gt_prefix + '_' + prediction_type)[prediction_type + '_rank'].mean().mean() * 100.0 / 250.0
            if verbose:
                print('per-class-mr', prediction_type, '{:2.2f}'.format(mu))
    print()

    mu = df['triplet_top1'].mean() * 100.0
    if verbose:
        print('simple-average', 'triplet', '{:2.2f}'.format(mu))

    for prediction_type in all_prediction_types:
        mu = df.groupby(gt_prefix + '_' + prediction_type)['triplet_top1'].mean().mean() * 100
        if verbose:
            print('per-class-average', 'triplet_' + prediction_type, '{:2.2f}'.format(mu))

    print()

    return collected_simple_means, collected_per_class_means



def get_group_counts(keys, ann_path):
    temp = pd.read_csv(ann_path).groupby(keys).size().reset_index(name='counts').sort_values('counts')
    temp = temp[keys + ['counts']]
    temp.index = pd.MultiIndex.from_arrays(temp[keys].values.T)
    return temp['counts']


def get_many_medium_few_scores(csv_path, cutoffs, data, data_dir, ann_dir, syn=True):
    df = pd.read_csv(csv_path)
    df['box_id'] = df.groupby('image_id').cumcount()
    metric_type = 'top1'
    all_prediction_types = ['rel', 'obj', 'sbj']
    if syn:
        if data == 'gvqa':
            syn_obj = pd.read_csv(data_dir + 'objects_synsets.csv')
            syn_obj = syn_obj[['object_name', 'synset']]
            syn_obj.set_index('object_name', inplace=True)

            syn_prd = pd.read_csv(data_dir + 'predicates_synsets.csv')
            syn_prd = syn_prd[['predicate_name', 'synset']]
            syn_prd.set_index('predicate_name', inplace=True)
        if data == 'vg8k':
            synsets = json.load(open(data_dir + 'words_synsets.json'))
            syn_obj = pd.DataFrame.from_dict(synsets['nouns'], orient='index', columns=['synset'])
            syn_prd = pd.DataFrame.from_dict(synsets['verbs'], orient='index', columns=['synset'])

    for prediction_type in all_prediction_types:
        df[prediction_type + '_' + metric_type] = df[prediction_type + '_rank'] < int(metric_type[3:])

    if syn:
        if data == 'gvqa':
            for prediction_type in ['sbj', 'obj']:
                df['gt_' + prediction_type + '_syn'] = syn_obj.loc[df['gt_' + prediction_type], 'synset'].to_list()
                df['det_' + prediction_type + '_syn'] = syn_obj.loc[df['det_' + prediction_type], 'synset'].to_list()
                df[prediction_type + '_top1_syn'] = df['gt_' + prediction_type + '_syn'] == df['det_' + prediction_type + '_syn']

            for prediction_type in ['rel']:
                df['gt_' + prediction_type + '_syn'] = syn_prd.loc[df['gt_' + prediction_type], 'synset'].to_list()
                df['det_' + prediction_type + '_syn'] = syn_prd.loc[df['det_' + prediction_type], 'synset'].to_list()
                df[prediction_type + '_top1_syn'] = df['gt_' + prediction_type + '_syn'] == df['det_' + prediction_type + '_syn']
        if data == 'vg8k':
            for prediction_type in ['sbj', 'obj']:
                df['gt_' + prediction_type + '_syn'] = syn_obj.reindex(df['gt_' + prediction_type])['synset'].to_list()
                df['det_' + prediction_type + '_syn'] = syn_obj.reindex(df['det_' + prediction_type])['synset'].to_list()
                df[prediction_type + '_top1_syn'] = df['gt_' + prediction_type + '_syn'] == df['det_' + prediction_type + '_syn']

            for prediction_type in ['rel']:
                df['gt_' + prediction_type + '_syn'] = syn_prd.reindex(df['gt_' + prediction_type])['synset'].to_list()
                df['det_' + prediction_type + '_syn'] = syn_prd.reindex(df['det_' + prediction_type])['synset'].to_list()
                df[prediction_type + '_top1_syn'] = df['gt_' + prediction_type + '_syn'] == df['det_' + prediction_type + '_syn']

    syn_key = ''
    if syn:
        syn_key = '_syn'

    df['triplet_top1' + syn_key] = df['rel_top1' + syn_key] & df['sbj_top1' + syn_key] & df['obj_top1' + syn_key]

    cutoff, cutoff_medium = cutoffs

    a = df.groupby('gt_rel').mean()
    classes_rel = (list(a.sort_values('rel_freq_gt').index))
    classes_rel_few = classes_rel[:int(len(classes_rel)*cutoff)]
    classes_rel_medium = classes_rel[int(len(classes_rel)*cutoff):int(len(classes_rel)*cutoff_medium)]
    classes_rel_many = classes_rel[int(len(classes_rel)*cutoff_medium):]

    a = df.groupby('gt_sbj').mean()
    classes_sbj = (list(a.sort_values('sbj_freq_gt').index))
    classes_sbj_few = classes_sbj[:int(len(classes_sbj)*cutoff)]
    classes_sbj_medium = classes_sbj[int(len(classes_sbj)*cutoff):int(len(classes_sbj)*cutoff_medium)]
    classes_sbj_many = classes_sbj[int(len(classes_sbj)*cutoff_medium):]

    a = df.groupby('gt_obj').mean()
    classes_obj = (list(a.sort_values('obj_freq_gt').index))
    classes_obj_few = classes_obj[:int(len(classes_obj)*cutoff)]
    classes_obj_medium = classes_obj[int(len(classes_obj)*cutoff):int(len(classes_obj)*cutoff_medium)]
    classes_obj_many = classes_obj[int(len(classes_obj)*cutoff_medium):]

    df_few_rel = df[df['gt_rel'].isin(classes_rel_few)]
    df_medium_rel = df[df['gt_rel'].isin(classes_rel_medium)]
    df_many_rel = df[df['gt_rel'].isin(classes_rel_many)]

    df_few_sbj = df[df['gt_sbj'].isin(classes_sbj_few)]
    df_medium_sbj = df[df['gt_sbj'].isin(classes_sbj_medium)]
    df_many_sbj = df[df['gt_sbj'].isin(classes_sbj_many)]

    df_few_obj = df[df['gt_obj'].isin(classes_obj_few)]
    df_medium_obj = df[df['gt_obj'].isin(classes_obj_medium)]
    df_many_obj = df[df['gt_obj'].isin(classes_obj_many)]


    # print('sbj_overall_top1', num(df_['sbj_top1'].mean() * 100.))
    # print('obj_overall_top1', num(df['obj_top1'].mean() * 100.))
    # print('rel few:', len(df_few_rel))
    # print('rel medium:',len(df_medium_rel))
    # print('rel many:', len(df_many_rel))
    #
    # print('sbj few:', len(df_few_sbj))
    # print('sbj medium:',len(df_medium_sbj))
    # print('sbj many:', len(df_many_sbj))
    #
    # print('obj few:', len(df_few_obj))
    # print('obj medium:',len(df_medium_obj))
    # print('obj many:', len(df_many_obj))
    # print('all:', len(df))
    # print()
    if syn:
        tables_title = 'synsets matching'
    else:
        tables_title = 'exact matching'

    print('=========================================================')
    print()

    print('Many, Medium, Few accuracy scores using {}:'.format(tables_title))

    print('rel many:', '{:2.2f}'.format(df_many_rel.groupby('gt_rel')['rel_top1' + syn_key].mean().mean() * 100.))
    print('rel med:', '{:2.2f}'.format(df_medium_rel.groupby('gt_rel')['rel_top1' + syn_key].mean().mean() * 100.))
    print('rel few:', '{:2.2f}'.format(df_few_rel.groupby('gt_rel')['rel_top1' + syn_key].mean().mean() * 100.))
    print('rel all (per-class):', '{:2.2f}'.format(df.groupby('gt_rel')['rel_top1' + syn_key].mean().mean() * 100.))
    print('rel all (per-example):', '{:2.2f}'.format(df['rel_top1' + syn_key].mean() * 100.))
    print()

    sbj_many = df_many_sbj.groupby('gt_sbj')['sbj_top1' + syn_key].mean().mean() * 100.
    sbj_med = df_medium_sbj.groupby('gt_sbj')['sbj_top1' + syn_key].mean().mean() * 100.
    sbj_few = df_few_sbj.groupby('gt_sbj')['sbj_top1' + syn_key].mean().mean() * 100.
    sbj_all = df.groupby('gt_sbj')['sbj_top1' + syn_key].mean().mean() * 100.
    sbj_all_o = df['sbj_top1'].mean() * 100.

    obj_many = df_many_obj.groupby('gt_obj')['obj_top1' + syn_key].mean().mean() * 100.
    obj_med = df_medium_obj.groupby('gt_obj')['obj_top1' + syn_key].mean().mean() * 100.
    obj_few = df_few_obj.groupby('gt_obj')['obj_top1' + syn_key].mean().mean() * 100.
    obj_all = df.groupby('gt_obj')['obj_top1' + syn_key].mean().mean() * 100.
    obj_all_o = df['obj_top1'].mean() * 100.

    print('sbj/obj many:', '{:2.2f}'.format((sbj_many + obj_many) / 2.))
    print('sbj/obj med:', '{:2.2f}'.format((sbj_med + obj_med) / 2.))
    print('sbj/obj few:', '{:2.2f}'.format((sbj_few + obj_few) / 2.))
    print('sbj/obj all (per-class):', '{:2.2f}'.format((sbj_all + obj_all) / 2.))
    print('sbj/obj all (per-example):', '{:2.2f}'.format((sbj_all_o + obj_all_o) / 2.))

    print('=========================================================')
    print()
    # print('triplet accuracy few:', df_few_rel['triplet_top1'].mean() * 100.)
    # print('triplet accuracy med:', df_medium_rel['triplet_top1'].mean() * 100.)
    # print('triplet accuracy man:', df_many_rel['triplet_top1'].mean() * 100.)
    # print('triplet accuracy all:', df['triplet_top1'].mean() * 100.)
    # print('=========================================================')

    # print('triplet accuracy few:', df_few_rel['triplet_top1_syn'].mean() * 100.)
    # print('triplet accuracy med:', df_medium_rel['triplet_top1_syn'].mean() * 100.)
    # print('triplet accuracy man:', df_many_rel['triplet_top1_syn'].mean() * 100.)
    # print('triplet accuracy all:', df['triplet_top1_syn'].mean() * 100.)
    # print('=========================================================')

    ann_path = ann_dir + 'rel_annotations_train.csv'

    def get_triplets_scores(groupby, ann_path, syn_key, count_suffix):
        groupby_ann = ['_'.join(s.split('_')[::-1]) for s in groupby]
        triplets_freqs = get_group_counts(groupby_ann, ann_path)
        triplets_freqs = triplets_freqs.reindex(df[groupby].to_records(index=False).tolist()).fillna(0)
        df['count' + count_suffix] = triplets_freqs.to_list()
        df_triplets = df.groupby(groupby).mean()[['triplet_top1' + syn_key, 'count' + count_suffix]]
        df_triplets = df_triplets.reset_index().sort_values(['count' + count_suffix], ascending=True)

        df_triplets_few = df_triplets.iloc[:int(cutoff * len(df_triplets))]
        df_triplets_medium = df_triplets.iloc[int(cutoff * len(df_triplets)):int(cutoff_medium * len(df_triplets))]
        df_triplets_many = df_triplets.iloc[int(cutoff_medium * len(df_triplets)):]

        triplet_score_few = df_triplets_few['triplet_top1' + syn_key].mean() * 100.
        triplet_score_medium = df_triplets_medium['triplet_top1' + syn_key].mean() * 100.
        triplet_score_many = df_triplets_many['triplet_top1' + syn_key].mean() * 100.
        triplet_score_all = df_triplets['triplet_top1' + syn_key].mean() * 100.
        return triplet_score_many, triplet_score_medium, triplet_score_few, triplet_score_all

    trip_so_scores_many, trip_so_scores_medium, trip_so_scores_few, trip_so_scores_all = get_triplets_scores(['gt_sbj', 'gt_obj'], ann_path, syn_key, '_so')
    trip_sr_scores_many, trip_sr_scores_medium, trip_sr_scores_few, trip_sr_scores_all = get_triplets_scores(['gt_sbj', 'gt_rel'], ann_path, syn_key, '_sr')
    trip_or_scores_many, trip_or_scores_medium, trip_or_scores_few, trip_or_scores_all = get_triplets_scores(['gt_obj', 'gt_rel'], ann_path, syn_key, '_or')
    trip_scores_many, trip_scores_medium, trip_scores_few, trip_scores_all = get_triplets_scores(['gt_sbj', 'gt_obj', 'gt_rel'], ann_path, syn_key, '')

    print('Triplet scores grouped by subject/object using {}:'.format(tables_title))

    print('triplet so many:', '{:2.2f}'.format(trip_so_scores_many))
    print('triplet so med:', '{:2.2f}'.format(trip_so_scores_medium))
    print('triplet so few:', '{:2.2f}'.format(trip_so_scores_few))
    print('triplet so all:', '{:2.2f}'.format(trip_so_scores_all))
    print()
    print('Triplet scores grouped by subject/relation using {}:'.format(tables_title))

    print('triplet sr many:', '{:2.2f}'.format(trip_sr_scores_many))
    print('triplet sr med:', '{:2.2f}'.format(trip_sr_scores_medium))
    print('triplet sr few:', '{:2.2f}'.format(trip_sr_scores_few))
    print('triplet sr all:', '{:2.2f}'.format(trip_sr_scores_all))
    print()
    print('Triplet scores grouped by object/relation using {}:'.format(tables_title))

    print('triplet or many:', '{:2.2f}'.format(trip_or_scores_many))
    print('triplet or med:', '{:2.2f}'.format(trip_or_scores_medium))
    print('triplet or few:', '{:2.2f}'.format(trip_or_scores_few))
    print('triplet or all:', '{:2.2f}'.format(trip_or_scores_all))
    print()
    print('Triplet scores grouped by subject/relation/object using {}:'.format(tables_title))

    print('triplet sro many:', '{:2.2f}'.format(trip_scores_many))
    print('triplet sro med:', '{:2.2f}'.format(trip_scores_medium))
    print('triplet sro few:', '{:2.2f}'.format(trip_scores_few))
    print('triplet sro all:', '{:2.2f}'.format(trip_scores_all))
    print('=========================================================')
    print()