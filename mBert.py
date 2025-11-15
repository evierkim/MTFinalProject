import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import sys
from datetime import datetime


class Output:
    def __init__(self, filename='output.log'):
        self.terminal = sys.stdout  # Save original stdout to avoid recursion
        self.log_file = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)  # Use original stdout, NOT print() to avoid recursion
        self.log_file.write(message)  # file
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


class mBertodel(nn.Module):
    def __init__(self, device='cpu'):
        super(mBertodel, self).__init__()
        
        self.device = device #should be cpu
        
        print("Loading multilingual BERT...")
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased').to(device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        #tokenizer converts text to numbesr

        for param in self.bert.parameters(): #this prevents the model from retraining
            param.requires_grad = False
        
        self.quality_predictor = nn.Sequential(
            nn.Linear(768 * 2, 256),  #
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 64),       # hidden layer
            nn.ReLU(), 
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)          # Direct to output
        )
        
        for module in self.quality_predictor:
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.5)  # Start in middle
    
    #text to vector
    def get_embeddings(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, 
                               truncation=True, max_length=128) #text to numbers, get pytorch tesnsors, add padding for same length, max token size 
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad(): #BERT embeddings
            outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :] #gets the first token which represents the entire vector/sentence
    
    def forward(self, ref_texts, cand_texts):
        ref_emb = self.get_embeddings(ref_texts) #reference to embedding
        cand_emb = self.get_embeddings(cand_texts) #candidate to embedding
        
        combined = torch.cat([ref_emb, cand_emb], dim=1) #combine embeddings
        
        logits = self.quality_predictor(combined)
        scores = torch.sigmoid(logits).view(-1)    #changes to [0,1]
        return scores


def load_data_check_quality(filename='wmt_mqm_large.tsv'):
    """Load and verify data quality"""
    print(f"Loading {filename}")
    df = pd.read_csv(filename, sep='\t')
    print(f"Loaded {len(df)} samples")
    
    # Score statistics
    scores = df['human_score'].values
    print(f"\nScore statistics:")
    print(f"  Min: {scores.min():.3f}")
    print(f"  Max: {scores.max():.3f}")
    print(f"  Mean: {scores.mean():.3f}")
    print(f"  Std: {scores.std():.3f}")
    print(f"  Unique: {len(np.unique(scores))}")

    # Create dataset
    dataset = []
    score_list = []
    
    for _, row in df.iterrows():
        if pd.isna(row['reference']) or pd.isna(row['candidate']): 
            continue #skip missing data
        
        dataset.append({ #convert data to strings
            'reference': str(row['reference']).strip(),
            'candidate': str(row['candidate']).strip()
        })
        score_list.append(float(row['human_score'])) #add float score to list
    
    return dataset, score_list


def train_with_monitoring(model, train_d, train_s, val_d, val_s, 
                          num_epochs=25, batch_size=8, lr=5e-5):
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01) #gradient descent, weigth decay for regularization (no overfitting)
    criterion = nn.MSELoss() #mean squared error
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5) #cut learning rate in half if no improvements for 3 epochs
    
    best_pearson = -1
    history = {'train_loss': [], 'val_pearson': [], 'val_pred_std': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        model.train() #has dropout
        total_loss = 0
        indices = list(range(len(train_d))) #shuffle data
        np.random.shuffle(indices)
        
        num_batches = (len(indices) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Training"): #gives progress bar.
            batch_idx = indices[i*batch_size:(i+1)*batch_size]

            #convert data into tensor
            batch_data = [train_d[idx] for idx in batch_idx] 
            batch_scores = torch.tensor([train_s[idx] for idx in batch_idx], 
                                       device=model.device, dtype=torch.float32)
            #reset gradients
            optimizer.zero_grad()
            
            refs = [d['reference'] for d in batch_data]
            cands = [d['candidate'] for d in batch_data]
            
            predictions = model(refs, cands) 
            
            loss = criterion(predictions, batch_scores) #predictions vs human scores...
            loss.backward() #calculate gradients
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step() #update weights
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        history['train_loss'].append(avg_loss)
        
        # Validation
        model.eval()
        val_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(val_d), batch_size):
                batch = val_d[i:i+batch_size]
                refs = [d['reference'] for d in batch]
                cands = [d['candidate'] for d in batch]
                
                predictions = model(refs, cands)
                val_predictions.extend(predictions.cpu().numpy())
        
        # Metrics
        if len(np.unique(val_predictions)) > 1: #looks for variance
            pearson = pearsonr(val_predictions, val_s)[0] #linear relationship (exact value), uses covariance adn standard deviation
            spearman = spearmanr(val_predictions, val_s)[0] #rank of translations (vector order correlation)
        else:
            pearson, spearman = 0.0, 0.0
        
        pred_std = np.std(val_predictions)
        pred_range = (np.min(val_predictions), np.max(val_predictions))
        
        history['val_pearson'].append(pearson)
        history['val_pred_std'].append(pred_std)
        
        print(f"Loss: {avg_loss:.4f}")
        print(f"Val Pearson: {pearson:.4f}, Spearman: {spearman:.4f}")
        print(f"Pred std: {pred_std:.4f}, range: [{pred_range[0]:.3f}, {pred_range[1]:.3f}]")
        
        scheduler.step(avg_loss) #adjusts learning rate
        
        if pearson > best_pearson:
            best_pearson = pearson
            best_state = model.state_dict().copy() #copy over the best model weights and store them in the best state
    
    return best_state, history


def main():
    sys.stdout = Output()
    print(f"{datetime.now()}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Load data
    dataset, scores = load_data_check_quality()
    
    if dataset is None:
        print("Data loading failed")
        return
    
    # Split
    n = len(dataset)
    train_idx = int(0.7 * n)
    val_idx = int(0.85 * n)
    
    train_d, train_s = dataset[:train_idx], scores[:train_idx]
    val_d, val_s = dataset[train_idx:val_idx], scores[train_idx:val_idx]
    test_d, test_s = dataset[val_idx:], scores[val_idx:]
    
    print(f"Split: Train={len(train_d)}, Val={len(val_d)}, Test={len(test_d)}\n")
    
    # Train
    model = mBertodel(device=device)
    
    print("training")
    
    best_state, history = train_with_monitoring(
        model, train_d, train_s, val_d, val_s,
        num_epochs=25, batch_size=8, lr=5e-5
    )
    #load best
    model.load_state_dict(best_state)
    

    print("\nEvaluation")
    model.eval()
    test_predictions = []
    
    with torch.no_grad():
        for i in range(0, len(test_d), 8):
            batch = test_d[i:i+8]
            refs = [d['reference'] for d in batch]
            cands = [d['candidate'] for d in batch]
            
            predictions = model(refs, cands)
            test_predictions.extend(predictions.cpu().numpy())
    
    test_pearson = pearsonr(test_predictions, test_s)[0]
    test_spearman = spearmanr(test_predictions, test_s)[0]
    
    print(f"\nTest Pearson: {test_pearson:.4f}")
    print(f"Test Spearman: {test_spearman:.4f}")
    print(f"Pred std: {np.std(test_predictions):.4f}")
    print(f"Pred range: [{np.min(test_predictions):.3f}, {np.max(test_predictions):.3f}]")
    
    # Compare to BLEU
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU()
        bleu_scores = []
        for s in test_d:
            sc = bleu.corpus_score([s['candidate']], [[s['reference']]])
            bleu_scores.append(sc.score / 100)
        
        if len(np.unique(bleu_scores)) > 1:
            bleu_p = pearsonr(bleu_scores, test_s)[0]
            print(f"\nBLEU Pearson: {bleu_p:.4f}")
            print(f"Improvement: {((test_pearson - bleu_p) / abs(bleu_p) * 100):+.1f}%")
    except:
        pass
    
    torch.save({'model': best_state, 'history': history}, 'mqm_model.pth')
    print("\nSaved to mqm_model.pth")

if __name__ == "__main__":
    main()