import json



if __name__ == '__main__':
    # root = 'data/gvqa/seed0/gvqa_test_annos.json'
    root = 'data/gvqa/seed0/rel_annotations_test.json'
    
    num_results = 0
    
    with open(root, 'r') as f:
        annos = json.load(f)
        
    for key, value in annos.items():
        num_results += len(value)
        
    print(num_results)