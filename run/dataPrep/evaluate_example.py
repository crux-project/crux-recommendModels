from ranx import Qrels, Run
from ranx import evaluate

qrels_dict = { "q_1": { '12': 5, '25': 3 },
               "q_2": { '11': 6, '22': 1 } }

run_dict = { "q_1": { '12': 0.9, '23': 0.8, '25': 0.7,
                      '36': 0.6, '32': 0.5, '35': 0.4  },
             "q_2": { '12': 0.9, '11': 0.8, '25': 0.7,
                      '36': 0.6, '22': 0.5, '35': 0.4  } }

qrels = Qrels(qrels_dict)
run = Run(run_dict)

# Compute score for a single metric
results=evaluate(qrels, run, "ndcg@5")
print(results)

multi_results=evaluate(qrels, run, ["map@5", "mrr"])
print(multi_results)