# wmt_evaluation.py
"""
WMT Dataset Evaluation for Enhanced BERTScore
Evaluates the enhanced BERTScore metric with synonym smoothing on the WMT MQM dataset.
Computes Pearson and Spearman correlation with human judgments and compares against baselines.
"""

import os
import pandas as pd
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sacrebleu import corpus_bleu
from bert_score import score as bert_score_func
import warnings
warnings.filterwarnings('ignore')

# Import the enhanced metric from extended_smartBoost
try:
    from extended_smartBoost import TokenLevelBERTScore
except ImportError:
    print("Error: Could not import TokenLevelBERTScore from extended_smartBoost.py")
    print("Make sure extended_smartBoost.py is in the same directory")
    exit(1)

class WMTEvaluator:
    def __init__(self, model_type='bert-base-uncased', device='cpu', multilingual=False):
        """Initialize the WMT evaluator with enhanced BERTScore.
        
        Args:
            model_type: BERT model to use
            device: CPU or CUDA device
            multilingual: If True, use multilingual BERT and don't filter by language
        """
        print("Initializing WMT Evaluator...")
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.multilingual = multilingual
        print(f"Using device: {self.device}")
        print(f"Multilingual mode: {multilingual}")
        
        # Choose model based on multilingual setting
        if multilingual:
            # Use multilingual BERT for cross-lingual evaluation
            model_type = 'bert-base-multilingual-cased'
            print("Using multilingual BERT for cross-lingual evaluation")
        
        # Initialize enhanced BERTScore with best parameters from corrected_results.json
        self.enhanced_scorer = TokenLevelBERTScore(model_type=model_type, device=self.device)
        
        # Best parameters from corrected_results.json
        self.best_params = {
            'beta': 3.0,
            'boost_threshold': 0.65,
            'alpha': 0.9,
            'synonym_threshold': 0.2,
            'method': 'combined'
        }
        
        print(f"Using enhanced BERTScore with {self.best_params['method']} smoothing")
    
    def detect_language(self, text):
        """Detect language of text (simple heuristic for common languages)."""
        # Simple language detection based on character sets
        text = str(text).lower()
        
        # Check for Cyrillic characters
        if any('\u0400' <= c <= '\u04FF' for c in text):
            return 'ru'  # Russian
        
        # Check for German umlauts and ß
        if any(c in 'äöüß' for c in text):
            return 'de'  # German
        
        # Check for French accents
        if any(c in 'àâæçéèêëîïôœùûüÿ' for c in text):
            return 'fr'  # French
        
        # Check for Spanish accents
        if any(c in 'áéíñóúü' for c in text):
            return 'es'  # Spanish
        
        # Default to English
        return 'en'
    
    def filter_english_pairs(self, df):
        """Filter dataset to keep only English reference-candidate pairs."""
        print("\nFiltering for English-only sentence pairs...")
        
        english_indices = []
        non_english_pairs = 0
        mixed_language_pairs = 0
        
        for idx, row in df.iterrows():
            ref_lang = self.detect_language(row['reference'])
            cand_lang = self.detect_language(row['candidate'])
            
            if ref_lang == 'en' and cand_lang == 'en':
                english_indices.append(idx)
            elif ref_lang != 'en' or cand_lang != 'en':
                non_english_pairs += 1
            if ref_lang != cand_lang:
                mixed_language_pairs += 1
        
        print(f"  Found {len(english_indices)} English-only pairs")
        print(f"  {non_english_pairs} pairs contain non-English text")
        print(f"  {mixed_language_pairs} pairs have different languages")
        
        if len(english_indices) == 0:
            print("Warning: No English-only pairs found!")
            print("This suggests the dataset may not contain English translations.")
            print("Consider using multilingual mode or checking your data.")
            return df  # Return all data as fallback
        
        return df.iloc[english_indices]
    
    def load_wmt_data(self, filepath='wmt_mqm_large.tsv', sample_size=None, english_only=True):
        """Load and preprocess WMT MQM dataset."""
        print(f"\nLoading WMT dataset from {filepath}...")
        
        try:
            # Load the TSV file
            df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
            
            print(f"Original dataset size: {len(df)}")
            print(f"Columns found: {df.columns.tolist()}")
            
            # Check for required columns - your file has 'human_score' not 'score'
            required_ref_candidate = ['reference', 'candidate']
            score_column = None
            
            # Look for score column
            for col in df.columns:
                if 'score' in col.lower() or 'human' in col.lower():
                    score_column = col
                    break
            
            if score_column is None:
                raise ValueError("No score column found in dataset")
            
            print(f"Using '{score_column}' as score column")
            
            # Check for reference and candidate columns
            missing_cols = [col for col in required_ref_candidate if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns {missing_cols}")
                print("Trying to find alternative column names...")
                
                # Try to infer columns
                column_map = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if 'ref' in col_lower:
                        column_map['reference'] = col
                    elif 'cand' in col_lower or 'mt' in col_lower or 'translation' in col_lower:
                        column_map['candidate'] = col
                
                if 'reference' not in column_map and len(df.columns) >= 2:
                    column_map['reference'] = df.columns[0]
                if 'candidate' not in column_map and len(df.columns) >= 3:
                    column_map['candidate'] = df.columns[1]
                
                if len(column_map) >= 2:
                    df = df.rename(columns=column_map)
                    print(f"Renamed columns: {column_map}")
                else:
                    raise ValueError(f"Could not identify required columns. Available: {df.columns.tolist()}")
            
            # Rename score column to 'score' for consistency
            if score_column != 'score':
                df = df.rename(columns={score_column: 'score'})
                print(f"Renamed '{score_column}' to 'score'")
            
            # Filter for English-only pairs if requested and not in multilingual mode
            if english_only and not self.multilingual:
                df = self.filter_english_pairs(df)
            
            # Clean and filter data
            print(f"\nData cleaning...")
            
            # Remove rows with NaN values
            initial_count = len(df)
            df = df.dropna(subset=['reference', 'candidate', 'score'])
            print(f"  After dropping NaN: {len(df)} rows (removed {initial_count - len(df)})")
            
            # Convert score to float and normalize if needed
            df['score'] = pd.to_numeric(df['score'], errors='coerce')
            df = df.dropna(subset=['score'])
            print(f"  After converting scores to numeric: {len(df)} rows")
            
            # Check score range
            score_min = df['score'].min()
            score_max = df['score'].max()
            print(f"  Score range: [{score_min:.4f}, {score_max:.4f}]")
            
            # Normalize scores to 0-1 range if needed
            if score_max > 1.0 or score_min < 0:
                print(f"  Normalizing scores to [0, 1] range...")
                df['score'] = (df['score'] - score_min) / (score_max - score_min)
                print(f"  New score range: [{df['score'].min():.4f}, {df['score'].max():.4f}]")
            
            # Filter out very short or long sentences
            df['ref_len'] = df['reference'].apply(lambda x: len(str(x).split()))
            df['cand_len'] = df['candidate'].apply(lambda x: len(str(x).split()))
            
            initial_count = len(df)
            df = df[(df['ref_len'] >= 3) & (df['ref_len'] <= 100) & 
                   (df['cand_len'] >= 3) & (df['cand_len'] <= 100)]
            filtered_count = initial_count - len(df)
            print(f"  Filtered out {filtered_count} sentences that were too short or too long")
            
            # Sample if requested
            if sample_size and sample_size < len(df):
                df = df.sample(sample_size, random_state=42)
                print(f"  Sampled {sample_size} instances for evaluation")
            
            print(f"\nFinal dataset size: {len(df)}")
            print(f"Score statistics: mean={df['score'].mean():.3f}, std={df['score'].std():.3f}")
            print(f"Score range: [{df['score'].min():.3f}, {df['score'].max():.3f}]")
            
            # Show language distribution
            if not self.multilingual:
                print(f"\nLanguage distribution in filtered data:")
                languages = []
                for idx, row in df.iterrows():
                    ref_lang = self.detect_language(row['reference'])
                    cand_lang = self.detect_language(row['candidate'])
                    languages.append((ref_lang, cand_lang))
                
                unique_pairs = set(languages)
                for lang_pair in unique_pairs:
                    count = languages.count(lang_pair)
                    print(f"  {lang_pair[0]}->{lang_pair[1]}: {count} pairs")
            
            # Show sample of data
            print(f"\nSample of data (first 3 rows):")
            for i in range(min(3, len(df))):
                ref = str(df.iloc[i]['reference'])
                cand = str(df.iloc[i]['candidate'])
                score = df.iloc[i]['score']
                ref_lang = self.detect_language(ref)
                cand_lang = self.detect_language(cand)
                print(f"  Row {i}: Score={score:.3f}, Languages: {ref_lang}->{cand_lang}")
                print(f"    Ref: {ref[:60]}..." if len(ref) > 60 else f"    Ref: {ref}")
                print(f"    Cand: {cand[:60]}..." if len(cand) > 60 else f"    Cand: {cand}")
            
            return df
            
        except Exception as e:
            print(f"Error loading WMT data: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def compute_bleu_scores(self, references, candidates):
        """Compute BLEU scores for translation pairs."""
        print("\nComputing BLEU scores...")
        
        # Convert to lists if they aren't already
        if isinstance(references, pd.Series):
            references = references.tolist()
        if isinstance(candidates, pd.Series):
            candidates = candidates.tolist()
        
        # Clean text for BLEU computation
        cleaned_refs = []
        cleaned_cands = []
        for ref, cand in zip(references, candidates):
            # Basic cleaning for BLEU
            cleaned_refs.append(str(ref).strip())
            cleaned_cands.append(str(cand).strip())
        
        # Compute corpus BLEU
        refs = [[ref] for ref in cleaned_refs]
        try:
            corpus_bleu_score = corpus_bleu(cleaned_cands, refs).score
            print(f"  Corpus BLEU: {corpus_bleu_score:.2f}")
        except Exception as e:
            print(f"  Warning: Could not compute corpus BLEU: {e}")
            corpus_bleu_score = 0.0
        
        # Compute sentence-level BLEU (approximate)
        sentence_bleus = []
        for ref, cand in zip(cleaned_refs, cleaned_cands):
            try:
                sent_bleu = corpus_bleu([cand], [[ref]]).score
                sentence_bleus.append(sent_bleu / 100.0)  # Normalize to 0-1
            except:
                sentence_bleus.append(0.0)
        
        print(f"  Sentence BLEU stats: mean={np.mean(sentence_bleus):.4f}, std={np.std(sentence_bleus):.4f}")
        
        return corpus_bleu_score, np.array(sentence_bleus)
    
    def compute_bert_scores(self, references, candidates, multilingual=False):
        """Compute original BERTScore for translation pairs."""
        print("\nComputing original BERTScore...")
        
        # Convert to lists if needed
        if isinstance(references, pd.Series):
            references = references.tolist()
        if isinstance(candidates, pd.Series):
            candidates = candidates.tolist()
        
        # Clean text
        cleaned_refs = [str(ref).strip() for ref in references]
        cleaned_cands = [str(cand).strip() for cand in candidates]
        
        # Take a smaller sample if we have many sentences to avoid memory issues
        max_samples = 200  # Limit for bert-score to avoid memory issues
        if len(cleaned_refs) > max_samples:
            print(f"  Taking first {max_samples} samples to avoid memory issues...")
            cleaned_refs = cleaned_refs[:max_samples]
            cleaned_cands = cleaned_cands[:max_samples]
        
        try:
            # Choose model based on multilingual setting
            model_type = 'bert-base-multilingual-cased' if multilingual else 'bert-base-uncased'
            
            # Use batch processing for efficiency
            P, R, F1 = bert_score_func(
                cleaned_cands, 
                cleaned_refs, 
                lang='en' if not multilingual else None,  # No language for multilingual
                model_type=model_type,
                verbose=True,
                device=self.device,
                batch_size=32  # Smaller batch size for memory
            )
            
            # Convert to numpy arrays
            precision_scores = P.numpy()
            recall_scores = R.numpy()
            f1_scores = F1.numpy()
            
            print(f"  BERTScore statistics - F1: mean={f1_scores.mean():.4f}, std={f1_scores.std():.4f}")
            print(f"  BERTScore range: [{f1_scores.min():.4f}, {f1_scores.max():.4f}]")
            
            return {
                'precision': precision_scores,
                'recall': recall_scores,
                'f1': f1_scores
            }
            
        except Exception as e:
            print(f"  Error computing BERTScore: {e}")
            print("  Falling back to token-level BERTScore without smoothing...")
            
            # Use our enhanced scorer without smoothing
            try:
                results = self.enhanced_scorer.compute_smoothed_bertscore(
                    cleaned_refs,
                    cleaned_cands,
                    smoothing_method='none'
                )
                
                f1_scores = results['f1']
                print(f"  Fallback BERTScore stats - F1: mean={np.mean(f1_scores):.4f}, std={np.std(f1_scores):.4f}")
                
                return {
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1': f1_scores
                }
            except Exception as e2:
                print(f"  Fallback also failed: {e2}")
                # Return dummy scores
                n_samples = len(cleaned_refs)
                dummy_scores = np.ones(n_samples) * 0.8
                return {
                    'precision': dummy_scores,
                    'recall': dummy_scores,
                    'f1': dummy_scores
                }
    
    def compute_enhanced_scores(self, references, candidates):
        """Compute enhanced BERTScore with synonym smoothing."""
        print("\nComputing enhanced BERTScore with synonym smoothing...")
        
        # Convert to lists
        if isinstance(references, pd.Series):
            ref_list = references.tolist()
        else:
            ref_list = references
            
        if isinstance(candidates, pd.Series):
            cand_list = candidates.tolist()
        else:
            cand_list = candidates
        
        # Clean text
        cleaned_refs = [str(ref).strip() for ref in ref_list]
        cleaned_cands = [str(cand).strip() for cand in cand_list]
        
        # Limit samples for performance
        max_samples = 200
        if len(cleaned_refs) > max_samples:
            print(f"  Taking first {max_samples} samples for enhanced scoring...")
            cleaned_refs = cleaned_refs[:max_samples]
            cleaned_cands = cleaned_cands[:max_samples]
        
        # Use best parameters from corrected_results.json
        try:
            results = self.enhanced_scorer.compute_smoothed_bertscore(
                cleaned_refs,
                cleaned_cands,
                smoothing_method=self.best_params['method'],
                beta=self.best_params['beta'],
                boost_threshold=self.best_params['boost_threshold'],
                alpha=self.best_params['alpha'],
                synonym_threshold=self.best_params['synonym_threshold']
            )
            
            f1_scores = results['f1']
            print(f"  Enhanced BERTScore statistics - F1: mean={np.mean(f1_scores):.4f}, std={np.std(f1_scores):.4f}")
            print(f"  Enhanced BERTScore range: [{np.min(f1_scores):.4f}, {np.max(f1_scores):.4f}]")
            
            return results
            
        except Exception as e:
            print(f"  Error computing enhanced BERTScore: {e}")
            # Return the same as regular BERTScore
            return self.compute_bert_scores(cleaned_refs, cleaned_cands, multilingual=self.multilingual)
    
    def compute_correlations(self, human_scores, metric_scores, metric_name):
        """Compute Pearson and Spearman correlations."""
        # Ensure they're numpy arrays
        human_arr = np.array(human_scores)
        metric_arr = np.array(metric_scores)
        
        # Ensure same length
        min_len = min(len(human_arr), len(metric_arr))
        human_arr = human_arr[:min_len]
        metric_arr = metric_arr[:min_len]
        
        # Remove any NaN values
        mask = ~(np.isnan(human_arr) | np.isnan(metric_arr))
        human_arr = human_arr[mask]
        metric_arr = metric_arr[mask]
        
        if len(human_arr) < 2:
            print(f"  Warning: Not enough valid samples for {metric_name}")
            return 0.0, 0.0, 0
        
        try:
            pearson_corr, pearson_p = pearsonr(human_arr, metric_arr)
            spearman_corr, spearman_p = spearmanr(human_arr, metric_arr)
            
            significance = ""
            if pearson_p < 0.01:
                significance = "**"
            elif pearson_p < 0.05:
                significance = "*"
            
            print(f"  {metric_name:25} Pearson: {pearson_corr:.4f}{significance} (p={pearson_p:.4e}) | "
                  f"Spearman: {spearman_corr:.4f} (p={spearman_p:.4e}) | N={len(human_arr)}")
            
            return pearson_corr, spearman_corr, len(human_arr)
        except Exception as e:
            print(f"  Error computing correlations for {metric_name}: {e}")
            return 0.0, 0.0, 0
    
    def analyze_top_cases(self, df, enhanced_scores, bert_scores, human_scores, n_cases=5):
        """Analyze cases where enhanced metric differs most from original BERTScore."""
        print(f"\n{'='*60}")
        print("ANALYZING TOP CASES")
        print(f"{'='*60}")
        
        # Ensure arrays have same length
        min_len = min(len(enhanced_scores), len(bert_scores), len(human_scores))
        enhanced_scores = enhanced_scores[:min_len]
        bert_scores = bert_scores[:min_len]
        human_scores = human_scores[:min_len]
        
        # Calculate differences
        differences = enhanced_scores - bert_scores
        
        analysis = {
            'improvements': [],
            'declines': [],
            'summary': {
                'mean_difference': float(np.mean(differences)),
                'std_difference': float(np.std(differences)),
                'max_improvement': float(np.max(differences)),
                'max_decline': float(np.min(differences)),
                'percent_improved': float((differences > 0).mean() * 100),
                'percent_worsened': float((differences < 0).mean() * 100)
            }
        }
        
        print(f"\nSummary Statistics:")
        print(f"  Mean difference (enhanced - original): {analysis['summary']['mean_difference']:.4f}")
        print(f"  Std of differences: {analysis['summary']['std_difference']:.4f}")
        print(f"  Maximum improvement: +{analysis['summary']['max_improvement']:.4f}")
        print(f"  Maximum decline: {analysis['summary']['max_decline']:.4f}")
        print(f"  Sentences improved: {analysis['summary']['percent_improved']:.1f}%")
        print(f"  Sentences worsened: {analysis['summary']['percent_worsened']:.1f}%")
        
        # Get indices of largest improvements and declines
        if len(differences) > 0:
            top_improvements_idx = np.argsort(differences)[-n_cases:][::-1]
            top_declines_idx = np.argsort(differences)[:n_cases]
            
            # Analyze top improvements
            print(f"\nTop {min(n_cases, len(top_improvements_idx))} improvements (enhanced > original):")
            for i, idx in enumerate(top_improvements_idx):
                if differences[idx] <= 0:
                    break
                    
                if idx < len(df):
                    ref_lang = self.detect_language(df.iloc[idx]['reference'])
                    cand_lang = self.detect_language(df.iloc[idx]['candidate'])
                    
                    case = {
                        'index': int(idx),
                        'reference': str(df.iloc[idx]['reference']),
                        'candidate': str(df.iloc[idx]['candidate']),
                        'human_score': float(human_scores[idx]),
                        'bert_score': float(bert_scores[idx]),
                        'enhanced_score': float(enhanced_scores[idx]),
                        'difference': float(differences[idx]),
                        'languages': f"{ref_lang}->{cand_lang}"
                    }
                    analysis['improvements'].append(case)
                    
                    print(f"\n{i+1}. Improvement: +{case['difference']:.4f} ({case['languages']})")
                    print(f"   Human score: {case['human_score']:.3f}")
                    print(f"   Original BERTScore: {case['bert_score']:.4f}")
                    print(f"   Enhanced BERTScore: {case['enhanced_score']:.4f}")
                    print(f"   Reference: {case['reference'][:80]}...")
                    print(f"   Candidate: {case['candidate'][:80]}...")
            
            # Analyze top declines
            print(f"\nTop {min(n_cases, len(top_declines_idx))} declines (enhanced < original):")
            for i, idx in enumerate(top_declines_idx):
                if differences[idx] >= 0:
                    break
                    
                if idx < len(df):
                    ref_lang = self.detect_language(df.iloc[idx]['reference'])
                    cand_lang = self.detect_language(df.iloc[idx]['candidate'])
                    
                    case = {
                        'index': int(idx),
                        'reference': str(df.iloc[idx]['reference']),
                        'candidate': str(df.iloc[idx]['candidate']),
                        'human_score': float(human_scores[idx]),
                        'bert_score': float(bert_scores[idx]),
                        'enhanced_score': float(enhanced_scores[idx]),
                        'difference': float(differences[idx]),
                        'languages': f"{ref_lang}->{cand_lang}"
                    }
                    analysis['declines'].append(case)
                    
                    print(f"\n{i+1}. Decline: {case['difference']:.4f} ({case['languages']})")
                    print(f"   Human score: {case['human_score']:.3f}")
                    print(f"   Original BERTScore: {case['bert_score']:.4f}")
                    print(f"   Enhanced BERTScore: {case['enhanced_score']:.4f}")
                    print(f"   Reference: {case['reference'][:80]}...")
                    print(f"   Candidate: {case['candidate'][:80]}...")
        
        return analysis
    
    def create_visualizations(self, results, output_dir='wmt_results'):
        """Create visualizations of the evaluation results."""
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 1. Correlation comparison bar chart
            plt.figure(figsize=(10, 6))
            
            metrics = ['BLEU', 'BERTScore', 'Enhanced BERTScore']
            pearson_values = [results['correlations']['bleu']['pearson'],
                             results['correlations']['bert']['pearson'],
                             results['correlations']['enhanced']['pearson']]
            spearman_values = [results['correlations']['bleu']['spearman'],
                              results['correlations']['bert']['spearman'],
                              results['correlations']['enhanced']['spearman']]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = plt.bar(x - width/2, pearson_values, width, label='Pearson', color='skyblue')
            bars2 = plt.bar(x + width/2, spearman_values, width, label='Spearman', color='lightcoral')
            
            plt.xlabel('Metric')
            plt.ylabel('Correlation with Human Judgments')
            title = 'WMT Evaluation: Correlation with Human Scores'
            if self.multilingual:
                title += ' (Multilingual)'
            else:
                title += ' (English-only)'
            plt.title(title)
            plt.xticks(x, metrics)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, (p, s) in enumerate(zip(pearson_values, spearman_values)):
                plt.text(i - width/2, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=9)
                plt.text(i + width/2, s + 0.01, f'{s:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/wmt_correlations.png', dpi=150)
            plt.close()
            print(f"[✓] Saved correlation comparison to {output_dir}/wmt_correlations.png")
            
            # 2. Score distribution comparison (only if we have enough data)
            if len(results['human_scores']) > 10:
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                plt.hist(results['human_scores'], bins=20, alpha=0.7, color='gray')
                plt.xlabel('Human Score')
                plt.ylabel('Frequency')
                plt.title('Human Score Distribution')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 3, 2)
                plt.hist(results['bert_scores'], bins=20, alpha=0.7, color='blue')
                plt.xlabel('BERTScore')
                plt.ylabel('Frequency')
                plt.title('BERTScore Distribution')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 3, 3)
                plt.hist(results['enhanced_scores'], bins=20, alpha=0.7, color='green')
                plt.xlabel('Enhanced BERTScore')
                plt.ylabel('Frequency')
                plt.title('Enhanced BERTScore Distribution')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/score_distributions.png', dpi=150)
                plt.close()
                print(f"[✓] Saved score distributions to {output_dir}/score_distributions.png")
            
            # 3. Scatter plots
            if len(results['human_scores']) > 10:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                
                # BLEU vs Human
                axes[0].scatter(results['human_scores'], results['bleu_scores'], alpha=0.5, s=10)
                axes[0].set_xlabel('Human Score')
                axes[0].set_ylabel('BLEU Score')
                axes[0].set_title(f'BLEU vs Human (r={results["correlations"]["bleu"]["pearson"]:.3f})')
                axes[0].grid(True, alpha=0.3)
                
                # BERTScore vs Human
                axes[1].scatter(results['human_scores'], results['bert_scores'], alpha=0.5, s=10, color='blue')
                axes[1].set_xlabel('Human Score')
                axes[1].set_ylabel('BERTScore')
                axes[1].set_title(f'BERTScore vs Human (r={results["correlations"]["bert"]["pearson"]:.3f})')
                axes[1].grid(True, alpha=0.3)
                
                # Enhanced vs Human
                axes[2].scatter(results['human_scores'], results['enhanced_scores'], alpha=0.5, s=10, color='green')
                axes[2].set_xlabel('Human Score')
                axes[2].set_ylabel('Enhanced BERTScore')
                axes[2].set_title(f'Enhanced vs Human (r={results["correlations"]["enhanced"]["pearson"]:.3f})')
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/scatter_plots.png', dpi=150)
                plt.close()
                print(f"[✓] Saved scatter plots to {output_dir}/scatter_plots.png")
            
            # 4. Difference analysis
            if 'analysis' in results and len(results['enhanced_scores']) > 10:
                differences = np.array(results['enhanced_scores']) - np.array(results['bert_scores'])
                
                plt.figure(figsize=(10, 6))
                plt.hist(differences, bins=20, alpha=0.7, color='purple', edgecolor='black')
                plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero difference')
                plt.axvline(x=np.mean(differences), color='green', linestyle='-', alpha=0.7, 
                           label=f'Mean: {np.mean(differences):.4f}')
                
                plt.xlabel('Difference (Enhanced - Original BERTScore)')
                plt.ylabel('Frequency')
                plt.title('Distribution of Score Differences')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/difference_distribution.png', dpi=150)
                plt.close()
                print(f"[✓] Saved difference distribution to {output_dir}/difference_distribution.png")
                
        except Exception as e:
            print(f"[!] Visualization error: {e}")
    
    def evaluate(self, wmt_file='wmt_mqm_large.tsv', sample_size=200, output_dir='wmt_results'):
        """Main evaluation function."""
        print(f"{'='*60}")
        print("WMT DATASET EVALUATION")
        if self.multilingual:
            print("(Multilingual Mode)")
        else:
            print("(English-only Mode)")
        print(f"{'='*60}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data (english_only=True unless in multilingual mode)
        df = self.load_wmt_data(wmt_file, sample_size=sample_size, english_only=not self.multilingual)
        
        if len(df) == 0:
            print("No data to evaluate!")
            return {}
        
        # Extract references, candidates, and human scores
        references = df['reference']
        candidates = df['candidate']
        human_scores = df['score'].values
        
        print(f"\nEvaluating {len(df)} translation pairs...")
        
        # Compute BLEU scores
        corpus_bleu, sentence_bleus = self.compute_bleu_scores(references, candidates)
        
        # Compute original BERTScore
        bert_results = self.compute_bert_scores(references, candidates, multilingual=self.multilingual)
        bert_scores = bert_results['f1']
        
        # Compute enhanced BERTScore
        enhanced_results = self.compute_enhanced_scores(references, candidates)
        enhanced_scores = enhanced_results['f1']
        
        # Compute correlations
        print(f"\n{'='*60}")
        print("CORRELATION WITH HUMAN JUDGMENTS")
        print(f"{'='*60}")
        
        # BLEU correlations
        bleu_pearson, bleu_spearman, bleu_n = self.compute_correlations(
            human_scores, sentence_bleus, "BLEU"
        )
        
        # BERTScore correlations
        bert_pearson, bert_spearman, bert_n = self.compute_correlations(
            human_scores, bert_scores, "Original BERTScore"
        )
        
        # Enhanced BERTScore correlations
        enhanced_pearson, enhanced_spearman, enhanced_n = self.compute_correlations(
            human_scores, enhanced_scores, "Enhanced BERTScore"
        )
        
        # Analyze cases
        analysis = self.analyze_top_cases(df, enhanced_scores, bert_scores, human_scores, n_cases=5)
        
        # Prepare results
        results = {
            'dataset_info': {
                'source_file': wmt_file,
                'sample_size': len(df),
                'multilingual': self.multilingual,
                'human_score_stats': {
                    'mean': float(np.mean(human_scores)),
                    'std': float(np.std(human_scores)),
                    'min': float(np.min(human_scores)),
                    'max': float(np.max(human_scores))
                }
            },
            'correlations': {
                'bleu': {
                    'pearson': float(bleu_pearson),
                    'spearman': float(bleu_spearman),
                    'n_samples': int(bleu_n),
                    'corpus_bleu': float(corpus_bleu)
                },
                'bert': {
                    'pearson': float(bert_pearson),
                    'spearman': float(bert_spearman),
                    'n_samples': int(bert_n)
                },
                'enhanced': {
                    'pearson': float(enhanced_pearson),
                    'spearman': float(enhanced_spearman),
                    'n_samples': int(enhanced_n),
                    'parameters': self.best_params
                }
            },
            'score_stats': {
                'bleu': {
                    'mean': float(np.mean(sentence_bleus)),
                    'std': float(np.std(sentence_bleus))
                },
                'bert': {
                    'mean': float(np.mean(bert_scores)),
                    'std': float(np.std(bert_scores))
                },
                'enhanced': {
                    'mean': float(np.mean(enhanced_scores)),
                    'std': float(np.std(enhanced_scores))
                }
            },
            'improvement_over_bert': {
                'pearson_delta': float(enhanced_pearson - bert_pearson),
                'spearman_delta': float(enhanced_spearman - bert_spearman)
            },
            'analysis': analysis,
            'human_scores': human_scores.tolist(),
            'bleu_scores': sentence_bleus.tolist(),
            'bert_scores': bert_scores.tolist(),
            'enhanced_scores': enhanced_scores.tolist()
        }
        
        # Calculate percentage improvements (avoid division by zero)
        if abs(bert_pearson) > 0.001:
            results['improvement_over_bert']['pearson_improvement_pct'] = \
                float((enhanced_pearson - bert_pearson) / abs(bert_pearson) * 100)
        else:
            results['improvement_over_bert']['pearson_improvement_pct'] = 0.0
            
        if abs(bert_spearman) > 0.001:
            results['improvement_over_bert']['spearman_improvement_pct'] = \
                float((enhanced_spearman - bert_spearman) / abs(bert_spearman) * 100)
        else:
            results['improvement_over_bert']['spearman_improvement_pct'] = 0.0
        
        # Save results
        results_file = f'{output_dir}/wmt_evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[✓] Results saved to {results_file}")
        
        # Create visualizations
        self.create_visualizations(results, output_dir)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        
        print(f"\nDataset: {wmt_file}")
        print(f"Sample size: {len(df)} sentences")
        print(f"Mode: {'Multilingual' if self.multilingual else 'English-only'}")
        print(f"\nCorrelation with Human Judgments:")
        print(f"{'Metric':25} {'Pearson':>10} {'Spearman':>10} {'Improvement':>12}")
        print(f"{'-'*60}")
        
        for name, data in results['correlations'].items():
            if name == 'bleu':
                metric_name = "BLEU"
                improvement = ""
            elif name == 'bert':
                metric_name = "Original BERTScore"
                improvement = ""
            else:
                metric_name = "Enhanced BERTScore"
                if 'improvement_over_bert' in results:
                    pearson_imp = results['improvement_over_bert'].get('pearson_improvement_pct', 0)
                    spearman_imp = results['improvement_over_bert'].get('spearman_improvement_pct', 0)
                    if pearson_imp > 0 or spearman_imp > 0:
                        improvement = f"+{pearson_imp:+.1f}%/{spearman_imp:+.1f}%"
                    else:
                        improvement = f"{pearson_imp:+.1f}%/{spearman_imp:+.1f}%"
                else:
                    improvement = ""
            
            print(f"{metric_name:25} {data['pearson']:10.4f} {data['spearman']:10.4f} {improvement:>12}")
        
        print(f"\nKey Findings:")
        
        if results['improvement_over_bert']['pearson_delta'] > 0:
            print(f"✓ Enhanced BERTScore improves Pearson correlation by "
                  f"{results['improvement_over_bert']['pearson_delta']:.4f}")
            if 'pearson_improvement_pct' in results['improvement_over_bert']:
                print(f"  (Relative improvement: +{results['improvement_over_bert']['pearson_improvement_pct']:.1f}%)")
        else:
            print(f"✗ Enhanced BERTScore decreases Pearson correlation by "
                  f"{abs(results['improvement_over_bert']['pearson_delta']):.4f}")
        
        if results['improvement_over_bert']['spearman_delta'] > 0:
            print(f"✓ Enhanced BERTScore improves Spearman correlation by "
                  f"{results['improvement_over_bert']['spearman_delta']:.4f}")
            if 'spearman_improvement_pct' in results['improvement_over_bert']:
                print(f"  (Relative improvement: +{results['improvement_over_bert']['spearman_improvement_pct']:.1f}%)")
        else:
            print(f"✗ Enhanced BERTScore decreases Spearman correlation by "
                  f"{abs(results['improvement_over_bert']['spearman_delta']):.4f}")
        
        if 'analysis' in results and 'summary' in results['analysis']:
            print(f"\n✓ {results['analysis']['summary']['percent_improved']:.1f}% of sentences received higher scores")
            print(f"✗ {results['analysis']['summary']['percent_worsened']:.1f}% of sentences received lower scores")
        
        if 'analysis' in results and 'improvements' in results['analysis'] and results['analysis']['improvements']:
            print(f"\nExample of successful improvement:")
            imp = results['analysis']['improvements'][0]
            print(f"  Human score: {imp['human_score']:.3f}")
            print(f"  Original BERTScore: {imp['bert_score']:.4f}")
            print(f"  Enhanced BERTScore: {imp['enhanced_score']:.4f} (+{imp['difference']:.4f})")
            print(f"  Languages: {imp.get('languages', 'Unknown')}")
            print(f"  Reference: {imp['reference'][:60]}...")
            print(f"  Candidate: {imp['candidate'][:60]}...")
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"{'='*60}")
        
        return results

def main():
    """Main function to run the evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Enhanced BERTScore on WMT Dataset')
    parser.add_argument('--wmt_file', default='wmt_mqm_large.tsv', 
                       help='Path to WMT MQM dataset file')
    parser.add_argument('--sample_size', type=int, default=200,
                       help='Number of samples to evaluate')
    parser.add_argument('--output_dir', default='wmt_results',
                       help='Directory to save results and visualizations')
    parser.add_argument('--model', default='bert-base-uncased',
                       help='BERT model to use for embeddings')
    parser.add_argument('--multilingual', action='store_true',
                       help='Use multilingual BERT and evaluate all language pairs')
    
    args = parser.parse_args()
    
    # Check if WMT file exists
    if not os.path.exists(args.wmt_file):
        print(f"Error: WMT file '{args.wmt_file}' not found!")
        print("\nCreating a sample file for testing...")
        create_sample_file()
    
    # Run evaluation
    evaluator = WMTEvaluator(model_type=args.model, multilingual=args.multilingual)
    results = evaluator.evaluate(
        wmt_file=args.wmt_file,
        sample_size=args.sample_size,
        output_dir=args.output_dir
    )
    
    # Print location of output files
    print(f"\nOutput files saved in '{args.output_dir}/':")
    print(f"1. wmt_evaluation_results.json - Complete evaluation results")
    print(f"2. wmt_correlations.png - Correlation comparison chart")
    print(f"3. score_distributions.png - Score distribution comparison")
    print(f"4. scatter_plots.png - Scatter plots vs human scores")
    print(f"5. difference_distribution.png - Distribution of score differences")

def analyze_score_distribution(self, scores):
    """Analyze the distribution of human scores."""
    print("\nAnalyzing human score distribution...")
    
    scores = np.array(scores)
    
    # Check if scores look like they might be inverted
    if np.mean(scores) > 0.8:  # Very high average
        print(f"  Warning: Scores are clustered high (mean={np.mean(scores):.3f})")
        print(f"  This suggests scores might be on a different scale")
    
    # Check for possible z-scores
    if np.std(scores) < 0.1:  # Very low variance
        print(f"  Warning: Scores have low variance (std={np.std(scores):.3f})")
        print(f"  This suggests scores might not be on 0-1 scale")
    
    # Check if scores might need inversion
    print(f"  Testing correlation direction...")
    
    # Show quartiles
    print(f"  Score quartiles:")
    print(f"    Q1 (25%): {np.percentile(scores, 25):.3f}")
    print(f"    Median:   {np.median(scores):.3f}")
    print(f"    Q3 (75%): {np.percentile(scores, 75):.3f}")
    
    # Check if most scores are above 0.5 (suspicious for 0-1 scale)
    above_half = np.sum(scores > 0.5) / len(scores) * 100
    print(f"  {above_half:.1f}% of scores are above 0.5")
    
    return scores

def create_sample_file():
    """Create a sample WMT file for testing."""
    print("\nCreating sample WMT data for testing...")
    
    sample_data = """reference	candidate	human_score
A cat sits on the mat.	A feline rests on the rug.	0.95
The weather is nice today.	It's beautiful outside.	0.85
He made a quick decision.	He took swift action.	0.90
She spoke with gentle persuasion.	She talked with mild coaxing.	0.88
The old man walked slowly.	The elderly gentleman strolled leisurely.	0.92
I love to read books.	I enjoy reading novels.	0.87
The car is very fast.	The automobile is extremely quick.	0.89
She has a beautiful smile.	Her grin is gorgeous.	0.86
The meeting was productive.	The conference yielded good results.	0.91
He is a talented musician.	He's a gifted artist.	0.93"""
    
    with open('wmt_mqm_large.tsv', 'w') as f:
        f.write(sample_data)
    
    print("Created sample wmt_mqm_large.tsv with 10 entries")
    print("Note: For real evaluation, you need the actual WMT dataset")

if __name__ == "__main__":
    main()