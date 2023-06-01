from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import heapq


def interaction_correlation(interaction1, interaction2, a):
    # Get dataset and model form interactions
    d1, m1 = interaction1
    d2, m2 = interaction2

    # Compute interaction correlation
    dataset_similarity = cosine_similarity([d1], [d2])[0][0]
    model_similarity = cosine_similarity([m1], [m2])[0][0]
    interaction_corr = dataset_similarity * a + model_similarity * (1-a)

    return interaction_corr


def top_models(nearest_data_ids, edges, data_embed, model_embed, num_edges_to_probe, target_embed):
    # Create graph
    graph = defaultdict(list)
    for dataset_id, model_id, performance in edges:
        if dataset_id in nearest_data_ids:
            graph[dataset_id].append((model_id, performance))

    # Compute scores for each model
    model_scores = defaultdict(float)
    for dataset_id, interactions in graph.items():
        for model_id, performance in interactions:
            interaction1 = [data_embed[dataset_id], model_embed[model_id]]
            interaction2 = [target_embed, model_embed[model_id]]
            interaction_corr = interaction_correlation(interaction1, interaction2, 0.5)
            model_scores[model_id] += interaction_corr * performance

    # Average the scores
    for model_id in model_scores:
        model_scores[model_id] /= len(graph)

    # Get top models with their scores
    top_models = heapq.nlargest(num_edges_to_probe, model_scores.items(), key=lambda x: x[1])

    return top_models

# In reg_parse_kaggle_new.py:
# Step 1: add "import probe_model"
# Step 2: comment l27: train_split=get_train_edges(nearest_data_ids, edges, data_embed, model_embed, num_edges_to_probe)
# Step 3: add 2 lines after commended l27:
#   train_split = probe_model.top_models(nearest_data_ids, edges, data_embed, model_embed, num_edges_to_probe, target_embed)
#   train_split = np.array([[0] + list(item) for item in train_split])
