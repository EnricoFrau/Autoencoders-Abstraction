import numpy as np
import torch
from AE.models import AE_0


def compute_all_decoded_features_dist_over_repetitions(model_path_kwargs, model_kwargs, repetitions=range(6), datasets=None):

    distances_over_repetitions = np.array([])

    for i in repetitions:
        model_path_kwargs['dataset'] = datasets[0]
        model_path_kwargs['train_num'] = i
        model_D = load_model(model_path_kwargs, model_kwargs)

        for j in repetitions:
            model_path_kwargs['dataset'] = datasets[1]
            model_path_kwargs['train_num'] = j
            model_D_prime = load_model(model_path_kwargs, model_kwargs)

            distances = compute_all_decoded_features_distances(
                compute_all_decoded_features(model_D), 
                compute_all_decoded_features(model_D_prime)
            )

            distances_over_repetitions = np.append(distances_over_repetitions, distances)

    return distances_over_repetitions



def compute_all_decoded_features_distances_without_repetitions(model_path_kwargs, model_kwargs, repetitions = (0,0), datasets=None):
    model_path_kwargs['dataset'] = datasets[0]
    model_path_kwargs['train_num'] = repetitions[0]
    model_D = load_model(model_path_kwargs, model_kwargs)

    model_path_kwargs['dataset'] = datasets[1]
    model_path_kwargs['train_num'] = repetitions[1]
    model_D_prime = load_model(model_path_kwargs, model_kwargs)

    distances = compute_all_decoded_features_distances(
        compute_all_decoded_features(model_D), 
        compute_all_decoded_features(model_D_prime)
    )
    return distances

def compute_all_decoded_features_distances(all_features_D, all_features_D_prime):
    distances = np.zeros(len(all_features_D))
    for i in range(len(all_features_D)):
            distances[i] = calc_single_decoded_feat_distance(all_features_D[i], all_features_D_prime[i])
    return distances



def compute_all_decoded_features(model):
    all_features = []
    for i in range(model.latent_dim):
        feature_i = get_feature_i(i, model)
        all_features.append(feature_i)
    return all_features



def calc_single_decoded_feat_distance(feature_i_D, feature_i_D_prime):
    return np.linalg.norm(feature_i_D - feature_i_D_prime)




def get_feature_i(i, model):
    model.eval()
    with torch.no_grad():
        ld = model.latent_dim
        input_feature_state = torch.zeros((1, ld))
        input_feature_state[0, i] = 1
        feature_i = model.decode(input_feature_state).cpu().numpy()
        feature_i = feature_i.squeeze(0)  # Remove the first dimension
    return feature_i



def load_model(model_path_kwargs, model_kwargs):
    my_model = AE_0(
        **model_kwargs
    ).to(model_kwargs['device'])
    model_path = f"../models/{model_path_kwargs['output_activation_encoder']}/{model_path_kwargs['train_type']}/{model_path_kwargs['latent_dim']}/{model_path_kwargs['dataset']}/dr{model_path_kwargs['decrease_rate']}_{model_path_kwargs['num_hidden_layers']}hl_{model_path_kwargs['train_num']}.pth"
    my_model.load_state_dict(torch.load(model_path, map_location=model_kwargs['device']))
    return my_model
