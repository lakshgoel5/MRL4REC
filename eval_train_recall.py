import os
import random
import torch
import numpy as np
from time import time
import heapq

from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test
from modules.LightGCN_mul import LightGCN
from modules.LightGCN_sin import LightGCN as LightGCN_sin

def test_train_exclusion_free(model, user_dict, n_params, Ks=[10, 20, 50], sample_users=None):
    """
    Evaluates recall on the training set interactions directly (candidate exclusion-free).
    For each user, candidate pool is ALL items [0, n_items).
    Ground truth is user_dict['train_user_set'][u].
    """
    model.eval()
    device = next(model.parameters()).device
    
    train_user_set = user_dict['train_user_set']
    n_items = n_params['n_items']
    
    users = list(train_user_set.keys())
    if sample_users is not None and sample_users < len(users):
        random.seed(42)
        users = random.sample(users, sample_users)
        
    n_users_eval = len(users)
    
    user_gcn_emb, item_gcn_emb = model.generate()
    
    recalls = {k: 0.0 for k in Ks}
    ndcgs = {k: 0.0 for k in Ks}
    
    batch_size = 1024
    K_max = max(Ks)
    
    with torch.no_grad():
        for start_idx in range(0, n_users_eval, batch_size):
            end_idx = min(start_idx + batch_size, n_users_eval)
            batch_users = users[start_idx:end_idx]
            
            u_batch = torch.LongTensor(batch_users).to(device)
            u_emb = user_gcn_emb[u_batch] # [batch, dim]
            
            # Rating matrix: [batch_users, n_items]
            # Batch item rating computation to prevent OOM
            scores_list = []
            i_batch_size = 4096
            for i_start in range(0, n_items, i_batch_size):
                i_end = min(i_start + i_batch_size, n_items)
                i_batch = torch.LongTensor(list(range(i_start, i_end))).to(device)
                i_emb = item_gcn_emb[i_batch]
                
                # Rating shape: [len(u_batch), i_end - i_start]
                rates = model.rating(u_emb, i_emb).detach().cpu()
                scores_list.append(rates)
                
            scores = torch.cat(scores_list, dim=1).numpy() # [len(batch_users), n_items]
            
            for idx, u in enumerate(batch_users):
                pos_train = set(train_user_set[u])
                if not pos_train:
                    continue
                    
                user_scores = scores[idx]
                topk_items = heapq.nlargest(K_max, range(n_items), key=lambda i: user_scores[i])
                
                r = [1 if item in pos_train else 0 for item in topk_items]
                
                for k in Ks:
                    hits = sum(r[:k])
                    recalls[k] += hits / float(len(pos_train))
                    
                    # NDCG@k
                    dcg = sum([r[i] / np.log2(i + 2) for i in range(min(k, len(r)))])
                    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(k, len(pos_train)))])
                    ndcgs[k] += (dcg / idcg) if idcg > 0 else 0.0

    avg_recalls = {k: recalls[k] / n_users_eval for k in Ks}
    avg_ndcgs = {k: ndcgs[k] / n_users_eval for k in Ks}
    return avg_recalls, avg_ndcgs

if __name__ == '__main__':
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}") if (args.cuda and torch.cuda.is_available()) else torch.device("cpu")
    
    print(f"=== Running Diagnostic Test on dataset: {args.dataset} ===")
    train_cf, user_dict, n_params, norm_mat = load_data(args)
    
    if args.loss == 'mrl' and hasattr(args, 'K') and args.K > 1:
        model = LightGCN(n_params, args, norm_mat).to(device)
    else:
        try:
            model = LightGCN(n_params, args, norm_mat).to(device)
        except Exception:
            model = LightGCN_sin(n_params, args, norm_mat).to(device)

    # Train for a few epochs or user specified epochs
    num_epochs = args.epoch if args.epoch > 0 else 10
    print(f"\n--- Training for {num_epochs} epochs to inspect Train vs Valid Recall ---")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_cf_tensor = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    
    K_val = getattr(args, 'K', 1)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        index = np.arange(len(train_cf_tensor))
        np.random.shuffle(index)
        train_cf_shuffled = train_cf_tensor[index].to(device)
        
        s = 0
        total_loss = 0
        while s + args.batch_size <= len(train_cf_tensor):
            batch_pairs = train_cf_shuffled[s : s + args.batch_size]
            
            # Neg sampling
            neg_items = []
            for u, _ in batch_pairs.cpu().numpy():
                u = int(u)
                negs = []
                for _ in range(args.n_negs * K_val):
                    while True:
                        neg = random.choice(range(n_params['n_items']))
                        if neg not in user_dict['train_user_set'][u]:
                            break
                    negs.append(neg)
                neg_items.append(negs)
                
            batch_dict = {
                'users': batch_pairs[:, 0],
                'pos_items': batch_pairs[:, 1],
                'neg_items': torch.LongTensor(neg_items).to(device)
            }
            
            loss, _, _ = model(epoch, batch_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            s += args.batch_size
            
        print(f"Epoch {epoch}/{num_epochs} - Loss: {total_loss:.4f}")
        
        if epoch % 5 == 0 or epoch == num_epochs:
            print("\n--- Evaluating Train-set Recall (Exclusion-Free) vs Valid-set Recall ---")
            train_recalls, train_ndcgs = test_train_exclusion_free(model, user_dict, n_params, Ks=[10, 20, 50], sample_users=1000)
            
            print(f"[TRAIN SET RECALL (Candidate Pool = All {n_params['n_items']} items)]")
            print(f"  Recall@10: {train_recalls[10]:.4f} | Recall@20: {train_recalls[20]:.4f} | Recall@50: {train_recalls[50]:.4f}")
            print(f"  NDCG@10:   {train_ndcgs[10]:.4f} | NDCG@20:   {train_ndcgs[20]:.4f} | NDCG@50:   {train_ndcgs[50]:.4f}")
            
            # Standard validation evaluation
            try:
                valid_ret = test(model, user_dict, n_params, mode='valid')
                print(f"[VALID SET RECALL (Standard Evaluation)]")
                print(f"  Recall@10: {valid_ret['recall'][0]:.4f} | Recall@20: {valid_ret['recall'][1]:.4f} | Recall@50: {valid_ret['recall'][2]:.4f}")
            except Exception as e:
                print(f"Valid set eval error: {e}")
                
            print(f"Random Baseline Recall@20 (Theoretical Expectation): ~{20.0 / n_params['n_items']:.6f}\n")
