# This is the main script that runs the whole pipeline.
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)


# 1) preprocessing
print('Preprocessing...')
if config['dataset']['name'] == 'Diabetics':
    import preprocess_diabetics

elif config['dataset']['name'] == 'MIMIC':
    import preprocess_MIMIC

elif config['dataset']['name'] == 'UTI':
    import preprocess_UTI

print('Preprocessing Done!')


if 'SemanticGroupNet' in config['algorithms']['names']:
    print('Semantic grouping...')
    # 2) Semantic Grouping
    import semantic_grouping
    print('Semantic grouping Done!')


# 3) Neural Network Training
print('Neural net training...')
import neuralnet_training
print('Neural net training Done!')


# 4) Prediction
print('Running predictions...')
import make_predictions
print('Predictions Done!')


# 5) Visualization 
print('Running visualizations...')
import make_visualizations
print('Visualizations Done!')

