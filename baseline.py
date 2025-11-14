import pandas as pd
import json
from sacrebleu import corpus_bleu
from bert_score import BERTScorer
import numpy as np
from tqdm import tqdm
import os

class BaselineEvaluator:
    def __init__(self, model_type='microsoft/deberta-xlarge-mnli'):
        print("Loading BERTScore...")
        self.bert_scorer = BERTScorer(model_type=model_type, lang="en")
        print("Baseline ready!")
    
    def compute_bleu(self, references, candidates):
        # sacreBLEU to compute BLEU
        refs = [[ref] for ref in references] if isinstance(references[0], str) else references
        bleu_score = corpus_bleu(candidates, refs)
        return bleu_score.score
    
    def compute_bertscore(self, references, candidates):
        P, R, F1 = self.bert_scorer.score(candidates, references)
        return {'precision': P.numpy(), 'recall': R.numpy(),'f1': F1.numpy()}
    
    def evaluate_baselines(self, test_suite_path, output_path):
        print(f"Loading test suite from {test_suite_path}")
        df = pd.read_csv(test_suite_path)
        references = df['reference'].tolist()
        candidates = df['candidate'].tolist()
        print("Computing BLEU...")
        bleu_score = self.compute_bleu(references, candidates)
        print("Computing BERTScore...")
        bert_scores = self.compute_bertscore(references, candidates)
        # avg BERTScore
        avg_bertscore_f1 = np.mean(bert_scores['f1'])
        results = {
            'bleu_score': float(bleu_score),
            'bertscore': {'precision': bert_scores['precision'].tolist(), 'recall': bert_scores['recall'].tolist(), 'f1': bert_scores['f1'].tolist(),'average_f1': float(avg_bertscore_f1)},
            'test_cases': [
                {'reference': ref, 'candidate': cand, 'bleu_score': 0.0, 'bertscore_f1': float(f1)}
                for ref, cand, f1 in zip(references, candidates, bert_scores['f1'])
            ]
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Baseline done!")
        print(f"BLEU: {bleu_score:.4f}")
        print(f"avg BERTScore: {avg_bertscore_f1:.4f}")
        print(f"Results put in {output_path}")
        return results

if __name__ == "__main__":
    # test
    evaluator = BaselineEvaluator()
    # test suite
    results = evaluator.evaluate_baselines(
        test_suite_path='data/test_suite.csv',
        output_path='results/baseline_results.json'
    )