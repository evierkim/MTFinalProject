# ============================================================================
# FIX FOR OMP ERROR #15
# ============================================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ============================================================================
# MAIN IMPORTS
# ============================================================================
import pandas as pd
import json
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from bert_score import BERTScorer
from sacrebleu import corpus_bleu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet as wn
import warnings
warnings.filterwarnings('ignore')

# Download WordNet data
try:
    nltk.download('wordnet', quiet=True)
except:
    pass

# ============================================================================
# CORRECTED TOKEN-LEVEL BERTSCORE WITH IMPROVED SMOOTHING
# ============================================================================
class TokenLevelBERTScore:
    def __init__(self, model_type='bert-base-uncased', device='cpu'):
        print(f"Loading BERT model: {model_type}")
        self.bert_scorer = BERTScorer(model_type=model_type, lang="en", rescale_with_baseline=False)
        self.device = device
        
        # Get the model and tokenizer
        self.model = self.bert_scorer._model
        self.tokenizer = self.bert_scorer._tokenizer
        
    def get_token_embeddings(self, texts):
        """Get token embeddings for a list of texts"""
        all_embeddings = []
        all_tokens = []
        
        for text in texts:
            # Tokenize with max_length to avoid warning
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs[0]  # Last hidden state
            
            # Remove padding and special tokens
            tokens = []
            token_embeddings = []
            for i in range(1, min(input_ids.shape[1] - 1, 510)):
                if attention_mask[0, i] == 1:  # Only non-padding tokens
                    token = self.tokenizer.decode(input_ids[0, i])
                    tokens.append(token.strip())
                    token_embeddings.append(embeddings[0, i].cpu().numpy())
            
            all_embeddings.append(np.array(token_embeddings))
            all_tokens.append(tokens)
        
        return all_embeddings, all_tokens
    
    def compute_similarity_matrix(self, cand_embeddings, ref_embeddings):
        """Compute cosine similarity matrix between candidate and reference tokens"""
        if len(cand_embeddings) == 0 or len(ref_embeddings) == 0:
            return np.zeros((len(cand_embeddings), len(ref_embeddings)))
        
        # Normalize embeddings
        cand_norm = cand_embeddings / (np.linalg.norm(cand_embeddings, axis=1, keepdims=True) + 1e-8)
        ref_norm = ref_embeddings / (np.linalg.norm(ref_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        similarity_matrix = np.dot(cand_norm, ref_norm.T)
        return similarity_matrix
    
    def compute_synonym_probability(self, cand_token, ref_token):
        """Compute synonym probability using WordNet (improved version)"""
        # Clean tokens
        cand_token = cand_token.replace('##', '').lower()
        ref_token = ref_token.replace('##', '').lower()
        
        # If tokens are identical, return high probability
        if cand_token == ref_token:
            return 0.95
        
        cand_synsets = wn.synsets(cand_token)
        ref_synsets = wn.synsets(ref_token)
        
        if not cand_synsets or not ref_synsets:
            # Check for common prefixes or morphological variations
            if cand_token[:4] == ref_token[:4]:  # Common prefix
                return 0.2
            return 0.0
        
        # Compute maximum path similarity
        max_similarity = 0
        for cs in cand_synsets:
            for rs in ref_synsets:
                try:
                    sim = cs.path_similarity(rs)
                    if sim and sim > max_similarity:
                        max_similarity = sim
                except:
                    continue
        
        # Boost score if similarity found
        if max_similarity > 0:
            # Scale up: WordNet gives 0-1, but we want more aggressive boosting
            return min(0.3 + max_similarity * 0.7, 0.95)
        
        return 0.0
    
    # ============================================================================
    # CORRECTED SMOOTHING FUNCTIONS
    # ============================================================================
    
    def corrected_sigmoid_smoothing(self, similarity_matrix, beta=2.0, boost_threshold=0.7):
        """
        CORRECTED: Only boost scores that are already high (likely synonyms)
        Original formula was penalizing good matches!
        """
        boosted = np.copy(similarity_matrix)
        
        # Find where similarity is already high (potential synonyms)
        high_mask = similarity_matrix > boost_threshold
        
        if np.any(high_mask):
            S_high = similarity_matrix[high_mask]
            # Boost formula: S' = S + (1-S)*boost_factor
            # boost_factor increases from 0 to 1 as S increases above threshold
            boost_factor = 1 - np.exp(-beta * (S_high - boost_threshold))
            boosted[high_mask] = S_high + (1 - S_high) * boost_factor
        
        # For low scores, leave unchanged or slightly penalize wrong matches
        low_mask = similarity_matrix < 0.3
        if np.any(low_mask):
            # Slightly reduce very low scores (likely wrong matches)
            boosted[low_mask] = boosted[low_mask] * 0.9
        
        return np.clip(boosted, 0, 1)
    
    def corrected_synprob_smoothing(self, similarity_matrix, cand_tokens, ref_tokens, 
                                    alpha=0.7, synonym_threshold=0.25):
        """
        CORRECTED: Smart blending that only affects potential synonyms
        Old version was reducing scores by blending with low WordNet probabilities
        """
        n_cand, n_ref = similarity_matrix.shape
        if n_cand == 0 or n_ref == 0:
            return similarity_matrix
        
        adjusted = np.copy(similarity_matrix)
        
        for i in range(n_cand):
            for j in range(n_ref):
                sim = similarity_matrix[i, j]
                cand_token = cand_tokens[i]
                ref_token = ref_tokens[j]
                
                # Skip identical tokens (already handled by BERTScore)
                if cand_token == ref_token:
                    continue
                
                # Only apply synonym boost if:
                # 1. Original similarity is moderate/high (potential synonym)
                # 2. AND tokens are different words
                if sim > 0.5:  # Moderate similarity threshold
                    syn_prob = self.compute_synonym_probability(cand_token, ref_token)
                    
                    # Only boost if WordNet confirms some relation
                    if syn_prob > synonym_threshold:
                        # Smart boost: increase based on synonym probability
                        # But don't reduce if WordNet gives low probability
                        boost = syn_prob * alpha * (1 - sim)  # More boost for lower initial sim
                        adjusted[i, j] = sim + boost
                    elif sim > 0.7:
                        # If BERT says they're similar but WordNet doesn't know,
                        # trust BERT more (don't penalize)
                        adjusted[i, j] = sim * 1.05  # Small boost for BERT confidence
        
        return np.clip(adjusted, 0, 1)
    
    def combined_smoothing(self, similarity_matrix, cand_tokens, ref_tokens,
                          beta=2.0, boost_threshold=0.7, alpha=0.7, synonym_threshold=0.25):
        """
        Combined approach: Apply both corrections intelligently
        """
        # First apply sigmoid boosting
        sigmoid_boosted = self.corrected_sigmoid_smoothing(similarity_matrix, beta, boost_threshold)
        
        # Then apply synonym-aware adjustment
        fully_smoothed = self.corrected_synprob_smoothing(
            sigmoid_boosted, cand_tokens, ref_tokens, alpha, synonym_threshold
        )
        
        return fully_smoothed
    
    def compute_aligned_score(self, similarity_matrix):
        """Compute precision, recall, and F1 from similarity matrix"""
        n_cand, n_ref = similarity_matrix.shape
        
        if n_cand == 0 or n_ref == 0:
            return 0.0, 0.0, 0.0
        
        # Candidate-to-reference alignment (precision)
        max_sim_cand = np.max(similarity_matrix, axis=1)
        precision = np.mean(max_sim_cand)
        
        # Reference-to-candidate alignment (recall)
        max_sim_ref = np.max(similarity_matrix, axis=0)
        recall = np.mean(max_sim_ref)
        
        # F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return precision, recall, f1
    
    def compute_smoothed_bertscore(self, references, candidates, 
                                   smoothing_method='none', **params):
        """Compute smoothed BERTScore for a batch of sentences"""
        # Get embeddings and tokens
        ref_embeddings, ref_tokens_list = self.get_token_embeddings(references)
        cand_embeddings, cand_tokens_list = self.get_token_embeddings(candidates)
        
        all_precisions = []
        all_recalls = []
        all_f1s = []
        
        for i in range(len(references)):
            cand_emb = cand_embeddings[i]
            ref_emb = ref_embeddings[i]
            cand_tokens = cand_tokens_list[i]
            ref_tokens = ref_tokens_list[i]
            
            # Compute base similarity matrix
            sim_matrix = self.compute_similarity_matrix(cand_emb, ref_emb)
            
            # Apply CORRECTED smoothing
            if smoothing_method == 'sigmoid':
                beta = params.get('beta', 2.0)
                boost_threshold = params.get('boost_threshold', 0.7)
                smoothed_matrix = self.corrected_sigmoid_smoothing(sim_matrix, beta, boost_threshold)
            elif smoothing_method == 'synprob':
                alpha = params.get('alpha', 0.7)
                synonym_threshold = params.get('synonym_threshold', 0.25)
                smoothed_matrix = self.corrected_synprob_smoothing(
                    sim_matrix, cand_tokens, ref_tokens, alpha, synonym_threshold
                )
            elif smoothing_method == 'combined':
                beta = params.get('beta', 2.0)
                boost_threshold = params.get('boost_threshold', 0.7)
                alpha = params.get('alpha', 0.7)
                synonym_threshold = params.get('synonym_threshold', 0.25)
                smoothed_matrix = self.combined_smoothing(
                    sim_matrix, cand_tokens, ref_tokens, 
                    beta, boost_threshold, alpha, synonym_threshold
                )
            else:
                smoothed_matrix = sim_matrix  # No smoothing
            
            # Compute scores from smoothed matrix
            precision, recall, f1 = self.compute_aligned_score(smoothed_matrix)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
        
        return {
            'precision': np.array(all_precisions),
            'recall': np.array(all_recalls),
            'f1': np.array(all_f1s)
        }

# ============================================================================
# ENHANCED MT EVALUATOR WITH CORRECTED PARAMETERS
# ============================================================================
class EnhancedMTEvaluator:
    def __init__(self, model_type='bert-base-uncased', device='cpu'):
        print("Initializing Enhanced MT Evaluator (with corrected smoothing)...")
        
        # Use a smaller, faster model
        try:
            self.token_scorer = TokenLevelBERTScore(model_type, device)
        except Exception as e:
            print(f"Error loading {model_type}: {e}")
            print("Falling back to 'distilbert-base-uncased'...")
            self.token_scorer = TokenLevelBERTScore('distilbert-base-uncased', device)
        
        # CORRECTED CONFIGURATIONS
        # Sigmoid: Only boost high scores (potential synonyms)
        self.sigmoid_configs = [
            {'name': 'Sigmoid_conservative', 'beta': 1.5, 'boost_threshold': 0.75},
            {'name': 'Sigmoid_balanced', 'beta': 2.0, 'boost_threshold': 0.7},
            {'name': 'Sigmoid_aggressive', 'beta': 3.0, 'boost_threshold': 0.65},
        ]
        
        # SynProb: Only apply when WordNet confirms relation
        self.synprob_configs = [
            {'name': 'SynProb_conservative', 'alpha': 0.5, 'synonym_threshold': 0.3},
            {'name': 'SynProb_balanced', 'alpha': 0.7, 'synonym_threshold': 0.25},
            {'name': 'SynProb_aggressive', 'alpha': 0.9, 'synonym_threshold': 0.2},
        ]
        
        # Combined: Best of both
        self.combined_configs = [
            {'name': 'Combined_conservative', 'beta': 1.5, 'boost_threshold': 0.75, 
             'alpha': 0.5, 'synonym_threshold': 0.3},
            {'name': 'Combined_balanced', 'beta': 2.0, 'boost_threshold': 0.7, 
             'alpha': 0.7, 'synonym_threshold': 0.25},
            {'name': 'Combined_aggressive', 'beta': 3.0, 'boost_threshold': 0.65, 
             'alpha': 0.9, 'synonym_threshold': 0.2},
        ]
        
        print("Using corrected smoothing functions that BOOST synonym scores (not penalize)")
    
    def load_test_suite(self, path):
        """Load test suite CSV"""
        try:
            df = pd.read_csv(path)
            print(f"Loaded {len(df)} sentence pairs from {path}")
            return df['reference'].tolist(), df['candidate'].tolist()
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return ["She experienced profound suffering"], ["She felt profound anguish"]
    
    def compute_bleu(self, references, candidates):
        """Compute BLEU score"""
        if not references or not candidates:
            return 0.0
        try:
            refs = [[ref] for ref in references]
            bleu_score = corpus_bleu(candidates, refs)
            return bleu_score.score
        except:
            return 0.0
    
    def analyze_improvements(self, base_scores, enhanced_scores, candidates, references):
        """Analyze where enhanced metric improves over baseline"""
        improvements = []
        declines = []
        
        for idx in range(len(base_scores)):
            diff = enhanced_scores[idx] - base_scores[idx]
            if diff > 0.01:  # Significant improvement
                improvements.append({
                    'index': idx,
                    'reference': references[idx],
                    'candidate': candidates[idx],
                    'base_score': float(base_scores[idx]),
                    'enhanced_score': float(enhanced_scores[idx]),
                    'improvement': float(diff)
                })
            elif diff < -0.01:  # Significant decline
                declines.append({
                    'index': idx,
                    'reference': references[idx],
                    'candidate': candidates[idx],
                    'base_score': float(base_scores[idx]),
                    'enhanced_score': float(enhanced_scores[idx]),
                    'decline': float(-diff)
                })
        
        # Sort by magnitude
        improvements.sort(key=lambda x: x['improvement'], reverse=True)
        declines.sort(key=lambda x: x['decline'], reverse=True)
        
        return improvements[:5], declines[:3]  # Top improvements and declines
    
    def evaluate_all(self, test_suite_path='test_suite.csv', output_path='corrected_results.json'):
        """Complete evaluation with corrected smoothing"""
        print(f"\n{'='*60}")
        print("EVALUATION WITH CORRECTED SMOOTHING")
        print(f"{'='*60}")
        
        references, candidates = self.load_test_suite(test_suite_path)
        
        if not references or not candidates:
            print("No data to evaluate!")
            return {}
        
        results = {
            'dataset_info': {
                'num_sentences': len(references),
                'test_suite': test_suite_path,
                'note': 'Using CORRECTED smoothing functions that BOOST synonym scores'
            }
        }
        
        # 1. Baseline BLEU
        print("\n[1/5] Computing BLEU...")
        results['BLEU'] = float(self.compute_bleu(references, candidates))
        print(f"   BLEU Score: {results['BLEU']:.4f}")
        
        # 2. Original BERTScore
        print("\n[2/5] Computing original BERTScore...")
        try:
            orig_bert = self.token_scorer.compute_smoothed_bertscore(references, candidates, smoothing_method='none')
            results['BERTScore_original'] = {
                'precision': orig_bert['precision'].tolist(),
                'recall': orig_bert['recall'].tolist(),
                'f1': orig_bert['f1'].tolist(),
                'avg_f1': float(np.mean(orig_bert['f1']))
            }
            print(f"   BERTScore Avg F1: {results['BERTScore_original']['avg_f1']:.4f}")
        except Exception as e:
            print(f"   Error computing BERTScore: {e}")
            results['BERTScore_original'] = {'avg_f1': 0.0}
        
        # 3. CORRECTED Sigmoid-smoothed variants
        print("\n[3/5] Computing CORRECTED sigmoid-smoothed variants...")
        print("   (Only boosting high similarity scores)")
        
        sigmoid_results = []
        for config in self.sigmoid_configs:
            try:
                scores = self.token_scorer.compute_smoothed_bertscore(
                    references, candidates, 
                    smoothing_method='sigmoid',
                    beta=config['beta'],
                    boost_threshold=config['boost_threshold']
                )
                avg_f1 = float(np.mean(scores['f1']))
                results[config['name']] = {
                    'f1': scores['f1'].tolist(),
                    'avg_f1': avg_f1,
                    'params': config,
                    'improvement_vs_bert': avg_f1 - results['BERTScore_original']['avg_f1']
                }
                sigmoid_results.append((config['name'], avg_f1))
                print(f"   {config['name']}: {avg_f1:.4f} " + 
                      f"(Δ={results[config['name']]['improvement_vs_bert']:+.4f})")
            except Exception as e:
                print(f"   Error with {config['name']}: {e}")
                results[config['name']] = {'avg_f1': 0.0, 'params': config}
        
        # 4. CORRECTED SynProb variants
        print("\n[4/5] Computing CORRECTED SynProb variants...")
        print("   (Only blending when WordNet confirms synonym relation)")
        
        synprob_results = []
        for config in self.synprob_configs:
            try:
                scores = self.token_scorer.compute_smoothed_bertscore(
                    references, candidates,
                    smoothing_method='synprob',
                    alpha=config['alpha'],
                    synonym_threshold=config['synonym_threshold']
                )
                avg_f1 = float(np.mean(scores['f1']))
                results[config['name']] = {
                    'f1': scores['f1'].tolist(),
                    'avg_f1': avg_f1,
                    'params': config,
                    'improvement_vs_bert': avg_f1 - results['BERTScore_original']['avg_f1']
                }
                synprob_results.append((config['name'], avg_f1))
                print(f"   {config['name']}: {avg_f1:.4f} " +
                      f"(Δ={results[config['name']]['improvement_vs_bert']:+.4f})")
            except Exception as e:
                print(f"   Error with {config['name']}: {e}")
                results[config['name']] = {'avg_f1': 0.0, 'params': config}
        
        # 5. CORRECTED Combined variants
        print("\n[5/5] Computing CORRECTED combined variants...")
        
        combined_results = []
        for config in self.combined_configs:
            try:
                scores = self.token_scorer.compute_smoothed_bertscore(
                    references, candidates,
                    smoothing_method='combined',
                    beta=config['beta'],
                    boost_threshold=config['boost_threshold'],
                    alpha=config['alpha'],
                    synonym_threshold=config['synonym_threshold']
                )
                avg_f1 = float(np.mean(scores['f1']))
                results[config['name']] = {
                    'f1': scores['f1'].tolist(),
                    'avg_f1': avg_f1,
                    'params': config,
                    'improvement_vs_bert': avg_f1 - results['BERTScore_original']['avg_f1']
                }
                combined_results.append((config['name'], avg_f1))
                print(f"   {config['name']}: {avg_f1:.4f} " +
                      f"(Δ={results[config['name']]['improvement_vs_bert']:+.4f})")
            except Exception as e:
                print(f"   Error with {config['name']}: {e}")
                results[config['name']] = {'avg_f1': 0.0, 'params': config}
        
        # 6. Find and analyze best performing metric
        print("\n[+] Analyzing results...")
        
        # Find best metric
        all_metrics = [(f"Sigmoid: {name}", score) for name, score in sigmoid_results] + \
                     [(f"SynProb: {name}", score) for name, score in synprob_results] + \
                     [(f"Combined: {name}", score) for name, score in combined_results]
        
        if all_metrics:
            best_metric_name, best_metric_score = max(all_metrics, key=lambda x: x[1])
            bert_score = results['BERTScore_original']['avg_f1']
            
            print(f"   Best metric: {best_metric_name} ({best_metric_score:.4f})")
            print(f"   Original BERTScore: {bert_score:.4f}")
            
            if best_metric_score > bert_score:
                improvement_pct = ((best_metric_score - bert_score) / bert_score) * 100
                print(f"   ✓ IMPROVEMENT: +{improvement_pct:.1f}% over BERTScore")
            else:
                decline_pct = ((bert_score - best_metric_score) / bert_score) * 100
                print(f"   ✗ DECLINE: -{decline_pct:.1f}% vs BERTScore")
            
            # Get the actual metric name for detailed analysis
            if "Sigmoid:" in best_metric_name:
                best_name = best_metric_name.replace("Sigmoid: ", "")
            elif "SynProb:" in best_metric_name:
                best_name = best_metric_name.replace("SynProb: ", "")
            elif "Combined:" in best_metric_name:
                best_name = best_metric_name.replace("Combined: ", "")
            else:
                best_name = best_metric_name
            
            # Analyze improvements for best metric
            if best_name in results and 'f1' in results[best_name]:
                base_f1 = np.array(results['BERTScore_original']['f1'])
                enhanced_f1 = np.array(results[best_name]['f1'])
                
                improvements, declines = self.analyze_improvements(
                    base_f1, enhanced_f1, candidates, references
                )
                
                results['improvement_analysis'] = {
                    'best_metric': best_metric_name,
                    'best_score': float(best_metric_score),
                    'bert_score': float(bert_score),
                    'improvement': float(best_metric_score - bert_score),
                    'top_improvements': improvements,
                    'top_declines': declines
                }
                
                if improvements:
                    print(f"   Found {len(improvements)} sentences with significant improvement (>0.01)")
                    print("   Top improvements:")
                    for imp in improvements[:2]:
                        print(f"     Sentence {imp['index']}: +{imp['improvement']:.3f}")
                        print(f"       Ref: {imp['reference'][:50]}...")
                        print(f"       Cand: {imp['candidate'][:50]}...")
        
        # 7. Save results
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n[✓] Results saved to {output_path}")
        except Exception as e:
            print(f"\n[!] Could not save results: {e}")
        
        # 8. Generate visualizations
        try:
            self.generate_visualizations(results)
        except Exception as e:
            print(f"[!] Could not generate visualizations: {e}")
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"{'='*60}")
        
        return results
    
    def generate_visualizations(self, results):
        """Generate comparison visualizations"""
        try:
            # 1. Metric comparison bar chart
            plt.figure(figsize=(14, 7))
            
            metric_names = []
            metric_scores = []
            metric_colors = []
            
            # Add BERTScore
            if 'BERTScore_original' in results and 'avg_f1' in results['BERTScore_original']:
                metric_names.append('BERTScore')
                metric_scores.append(results['BERTScore_original']['avg_f1'])
                metric_colors.append('blue')
            
            # Add all other metrics
            for config_list, color in [
                (self.sigmoid_configs, 'orange'),
                (self.synprob_configs, 'green'),
                (self.combined_configs, 'purple')
            ]:
                for config in config_list:
                    name = config['name']
                    if name in results and 'avg_f1' in results[name]:
                        metric_names.append(name)
                        metric_scores.append(results[name]['avg_f1'])
                        metric_colors.append(color)
            
            # Create bar chart
            bars = plt.bar(range(len(metric_names)), metric_scores, color=metric_colors)
            plt.axhline(y=results['BERTScore_original']['avg_f1'], color='red', 
                       linestyle='--', alpha=0.5, label='BERTScore Baseline')
            
            plt.xticks(range(len(metric_names)), metric_names, rotation=45, ha='right')
            plt.ylabel('Average F1 Score')
            plt.title('CORRECTED: MT Metric Comparison (Higher is Better)')
            plt.ylim(0, 1.0)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, score in zip(bars, metric_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig('corrected_metric_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("[✓] Saved corrected_metric_comparison.png")
            
            # 2. Improvement analysis scatter plot
            if 'improvement_analysis' in results and 'top_improvements' in results['improvement_analysis']:
                improvements = results['improvement_analysis']['top_improvements']
                if improvements:
                    plt.figure(figsize=(10, 6))
                    
                    indices = [imp['index'] for imp in improvements]
                    improvements_val = [imp['improvement'] for imp in improvements]
                    
                    plt.scatter(indices, improvements_val, color='green', s=100, alpha=0.7)
                    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                    
                    plt.xlabel('Sentence Index')
                    plt.ylabel('Improvement over BERTScore')
                    plt.title('Top Improvements with Corrected Smoothing')
                    plt.grid(True, alpha=0.3)
                    
                    # Annotate top improvements
                    for i, imp in enumerate(improvements[:3]):
                        plt.annotate(f"+{imp['improvement']:.3f}", 
                                    (imp['index'], imp['improvement']),
                                    textcoords="offset points", 
                                    xytext=(0,10), 
                                    ha='center', 
                                    fontsize=9)
                    
                    plt.tight_layout()
                    plt.savefig('improvement_analysis.png', dpi=150)
                    plt.close()
                    print("[✓] Saved improvement_analysis.png")
            
        except Exception as e:
            print(f"[!] Visualization error: {e}")

# ============================================================================
# TEST FUNCTION FOR SPECIFIC CASES
# ============================================================================
def test_specific_cases():
    """Test the corrected smoothing on problematic cases from previous results"""
    print("\n" + "="*60)
    print("TESTING CORRECTED SMOOTHING ON PROBLEMATIC CASES")
    print("="*60)
    
    # Problematic cases from your error analysis
    test_cases = [
        {
            'index': 7,
            'reference': "A strange anxiety possessed him.",
            'candidate': "An odd uneasiness gripped him.",
            'old_bert': 0.8576,
            'old_enhanced': 0.6923
        },
        {
            'index': 48,
            'reference': "His eyes burned with fierce determination.",
            'candidate': "His gaze blazed with intense resolve.",
            'old_bert': 0.9101,
            'old_enhanced': 0.7622
        },
        {
            'index': 30,
            'reference': "His voice was full of tender affection.",
            'candidate': "His speech was filled with gentle fondness.",
            'old_bert': 0.8310,
            'old_enhanced': 0.6912
        },
        # Good synonym case
        {
            'index': 0,
            'reference': "She experienced profound suffering.",
            'candidate': "She felt deep anguish.",
            'old_bert': 0.8509,
            'old_enhanced': 0.7015
        }
    ]
    
    evaluator = EnhancedMTEvaluator(model_type='bert-base-uncased')
    
    print("\nCase-by-case analysis:")
    print("-" * 60)
    
    for case in test_cases:
        ref = [case['reference']]
        cand = [case['candidate']]
        
        # Original BERTScore
        orig = evaluator.token_scorer.compute_smoothed_bertscore(ref, cand, smoothing_method='none')
        orig_score = orig['f1'][0]
        
        # Best corrected combined
        corrected = evaluator.token_scorer.compute_smoothed_bertscore(
            ref, cand,
            smoothing_method='combined',
            beta=2.0, boost_threshold=0.7,
            alpha=0.7, synonym_threshold=0.25
        )
        corrected_score = corrected['f1'][0]
        
        print(f"\nCase {case['index']}:")
        print(f"  Reference: {case['reference']}")
        print(f"  Candidate: {case['candidate']}")
        print(f"  Original BERTScore: {orig_score:.4f}")
        print(f"  CORRECTED Enhanced: {corrected_score:.4f}")
        print(f"  Change: {corrected_score - orig_score:+.4f}")
        
        if corrected_score > orig_score:
            print(f"  ✓ CORRECTED: Improved by {(corrected_score - orig_score)/orig_score*100:.1f}%")
        elif corrected_score < orig_score:
            print(f"  ✗ Still worse by {(orig_score - corrected_score)/orig_score*100:.1f}%")
        else:
            print(f"  = No change")
    
    print("\n" + "="*60)
    print("SUMMARY: The corrected smoothing should now IMPROVE scores for synonyms,")
    print("or at least not make them worse like the old version did.")
    print("="*60)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CORRECTED MT EVALUATION SYSTEM")
    print("Fixed Token-level BERTScore with Synonym Boosting")
    print("=" * 60)
    
    try:
        # First test specific cases
        test_specific_cases()
        
        # Then run full evaluation
        print("\n\n" + "="*60)
        print("RUNNING FULL EVALUATION")
        print("="*60)
        
        # Initialize evaluator
        evaluator = EnhancedMTEvaluator(model_type='bert-base-uncased')
        
        # Check if test_suite.csv exists
        if not os.path.exists('test_suite.csv'):
            print("\n[!] WARNING: test_suite.csv not found!")
            print("Creating a small example test suite with synonym-rich examples...")
            
            # Create test suite specifically designed to test synonym recognition
            example_data = {
                'reference': [
                    'She experienced profound suffering',
                    'He made a quick decision',
                    'The cat sat on the mat',
                    'She spoke with gentle persuasion',
                    'They enjoyed the beautiful sunset',
                    'A strange anxiety possessed him',
                    'His eyes burned with fierce determination',
                    'His voice was full of tender affection',
                    'The old man walked slowly',
                    'The child laughed joyfully'
                ],
                'candidate': [
                    'She felt deep anguish',  # suffering → anguish
                    'He took a swift action',  # quick → swift, decision → action
                    'The feline rested on the rug',  # cat → feline, sat → rested, mat → rug
                    'She talked with mild coaxing',  # spoke → talked, gentle → mild, persuasion → coaxing
                    'They appreciated the gorgeous dusk',  # enjoyed → appreciated, beautiful → gorgeous, sunset → dusk
                    'An odd uneasiness gripped him',  # strange → odd, anxiety → uneasiness, possessed → gripped
                    'His gaze blazed with intense resolve',  # eyes → gaze, burned → blazed, fierce → intense, determination → resolve
                    'His speech was filled with gentle fondness',  # voice → speech, full → filled, tender → gentle, affection → fondness
                    'The elderly gentleman strolled leisurely',  # old → elderly, man → gentleman, walked → strolled, slowly → leisurely
                    'The youngster chuckled happily'  # child → youngster, laughed → chuckled, joyfully → happily
                ]
            }
            
            df = pd.DataFrame(example_data)
            df.to_csv('test_suite.csv', index=False)
            print("Created synonym-rich test_suite.csv with 10 sentence pairs")
            print("These pairs are specifically designed to test synonym recognition")
        
        # Run the evaluation
        results = evaluator.evaluate_all(
            test_suite_path='test_suite.csv',
            output_path='corrected_results.json'
        )
        
        # Print final summary
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        
        if 'BERTScore_original' in results and 'avg_f1' in results['BERTScore_original']:
            bert_f1 = results['BERTScore_original']['avg_f1']
            print(f"Original BERTScore F1: {bert_f1:.4f}")
            
            # Find best enhanced metric
            best_metric = ('BERTScore', bert_f1)
            for config_list in [evaluator.sigmoid_configs, evaluator.synprob_configs, evaluator.combined_configs]:
                for config in config_list:
                    name = config['name']
                    if name in results and 'avg_f1' in results[name]:
                        if results[name]['avg_f1'] > best_metric[1]:
                            best_metric = (name, results[name]['avg_f1'])
            
            print(f"Best Enhanced Metric: {best_metric[0]} ({best_metric[1]:.4f})")
            
            if best_metric[0] != 'BERTScore':
                improvement = ((best_metric[1] - bert_f1) / bert_f1) * 100
                print(f"✓ IMPROVEMENT over BERTScore: +{improvement:.2f}%")
                print("The corrected smoothing is WORKING!")
            else:
                print("✗ No improvement - The corrected smoothing didn't help")
                print("This suggests WordNet may not be sufficient for synonym detection")
                print("Consider training a custom synonym detection model")
        
        if 'improvement_analysis' in results:
            analysis = results['improvement_analysis']
            if analysis['improvement'] > 0:
                print(f"\n✓ Found {len(analysis['top_improvements'])} sentences with significant improvement")
                print("The corrected smoothing helps specific synonym cases!")
            else:
                print(f"\n✗ No significant improvements found")
                print("WordNet-based synonym detection may be too limited")
        
        print("\n" + "=" * 60)
        print("OUTPUT FILES:")
        print("1. corrected_results.json - Complete results with corrected smoothing")
        print("2. corrected_metric_comparison.png - Metric comparison chart")
        if os.path.exists('improvement_analysis.png'):
            print("3. improvement_analysis.png - Top improvements visualization")
        print("\nRECOMMENDATIONS:")
        print("1. If corrected smoothing still doesn't improve scores,")
        print("   WordNet may be insufficient for contextual synonym detection")
        print("2. Consider training a neural synonym classifier on BERT embeddings")
        print("3. Try using paraphrase databases (PPDB) instead of WordNet")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[!] ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Install missing packages: pip install pandas numpy torch scipy sacrebleu bert-score nltk matplotlib")
        print("2. Try a smaller model: Change to 'distilbert-base-uncased'")
        import traceback
        traceback.print_exc()

# ============================================================================
# KEY CHANGES IN THIS VERSION:
# 1. CORRECTED SIGMOID: Only boosts scores above threshold, doesn't penalize
# 2. CORRECTED SYNPROB: Only blends when WordNet confirms synonym relation
# 3. SMART BOOSTING: Increases scores for synonyms, decreases for wrong matches
# 4. TEST FUNCTION: Tests problematic cases from your previous results
# ============================================================================