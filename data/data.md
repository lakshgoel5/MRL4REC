Edited data.md
Viewed data.md:41-60
Listed directory data
Listed directory phone
Listed directory tripartite_aug_ttv
Ran command: `python3 -c "
import os

def analyze_dataset(data_dir):
    print(f'================ Analysis for dataset: {data_dir} ================')
    splits = {}
    for split in ['train.txt', 'valid.txt', 'test.txt']:
        fp = os.path.join(data_dir, split)
        if not os.path.exists(fp):
            continue
        user_items = {}
        total_lines = 0
        total_interactions = 0
        with open(fp, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                user_id = int(parts[0])
                items = [int(x) for x in parts[1:]]
                user_items[user_id] = items
                total_lines += 1
                total_interactions += len(items)
        splits[split] = user_items
        print(f'Split {split}: {total_lines} users, {total_interactions} total interactions')
    
    if 'train.txt' in splits and 'test.txt' in splits:
        train_users = set(splits['train.txt'].keys())
        test_users = set(splits['test.txt'].keys())
        common_users = train_users.intersection(test_users)
        print(f'\nUsers analysis:')
        print(f'  Train users count: {len(train_users)}')
        print(f'  Test users count: {len(test_users)}')
        print(f'  Common users count: {len(common_users)} ({len(common_users)/len(test_users)*100:.2f}% of test users in train)')
        
        # Check item overlap per common user
        all_train_items = set(item for items in splits['train.txt'].values() for item in items)
        all_test_items = set(item for items in splits['test.txt'].values() for item in items)
        common_items_global = all_train_items.intersection(all_test_items)
        print(f'\nGlobal items analysis:')
        print(f'  Train unique items: {len(all_train_items)}')
        print(f'  Test unique items: {len(all_test_items)}')
        print(f'  Global Common items: {len(common_items_global)} ({len(common_items_global)/len(all_test_items)*100:.2f}% of test items in train)')
        
        # Per-user item overlap
        per_user_item_overlap = []
        user_train_len = []
        user_test_len = []
        for u in common_users:
            tr_items = set(splits['train.txt'][u])
            te_items = set(splits['test.txt'][u])
            overlap = tr_items.intersection(te_items)
            per_user_item_overlap.append(len(overlap))
            user_train_len.append(len(splits['train.txt'][u]))
            user_test_len.append(len(splits['test.txt'][u]))
            
        print(f'\nPer-User interaction distribution (for common users):')
        print(f'  Avg train items per user: {sum(user_train_len)/len(user_train_len):.2f} (min={min(user_train_len)}, max={max(user_train_len)})')
        print(f'  Avg test items per user: {sum(user_test_len)/len(user_test_len):.2f} (min={min(user_test_len)}, max={max(user_test_len)})')
        print(f'  Avg overlapping items per user in train & test: {sum(per_user_item_overlap)/len(per_user_item_overlap):.2f}')
        print(f'  Users with >0 overlapping items in train & test: {sum(1 for x in per_user_item_overlap if x > 0)} / {len(common_users)}')

mrl_data = '/Users/lakshgoel/Desktop/BTP/MRL4REC/data'
for d in sorted(os.listdir(mrl_data)):
    dp = os.path.join(mrl_data, d)
    if os.path.isdir(dp):
        analyze_dataset(dp)
        print('\n')
"`
Viewed ml_100k_random.ipynb:1-234
Viewed sport_5-random.ipynb:1-348
Ran command: `python3 -c "
import os

fp_tr = '/Users/lakshgoel/Desktop/BTP/MRL4REC/data/tripartite_aug_ttv/train.txt'
fp_va = '/Users/lakshgoel/Desktop/BTP/MRL4REC/data/tripartite_aug_ttv/valid.txt'
fp_te = '/Users/lakshgoel/Desktop/BTP/MRL4REC/data/tripartite_aug_ttv/test.txt'

def load_txt(fp):
    u_map = {}
    with open(fp, 'r') as f:
        for line in f:
            parts = [int(x) for x in line.strip().split()]
            u_map[parts[0]] = parts[1:]
    return u_map

tr = load_txt(fp_tr)
va = load_txt(fp_va)
te = load_txt(fp_te)

print('Train users range:', min(tr.keys()), 'to', max(tr.keys()))
print('Valid users range:', min(va.keys()), 'to', max(va.keys()))
print('Test users range:', min(te.keys()), 'to', max(te.keys()))
"`
Listed directory MRL4REC
Listed directory utils
Viewed data_loader.py:1-171

### 1. Analysis of `MRL4REC/data` Format

In `MRL4REC/data`, datasets (such as `phone`, `sport`, `tool`, `ml100k`) use a standard space-separated collaborative filtering text format (`train.txt`, `test.txt`, and optionally `valid.txt`).

#### Line Format
Each line in `train.txt` / `test.txt` represents one user and all items interacted with by that user:
```text
<user_id> <item_id_1> <item_id_2> <item_id_3> ... <item_id_k>
```
* **`user_id`**: Zero-based integer index ($0, 1, 2, \dots, N_{\text{users}}-1$).
* **`item_id`**: Space-separated zero-based integer indices of products interacted with by that user.

---

### 2. User & Product Distribution Across Train & Test in MRL4REC

Our empirical analysis of the benchmark datasets (`phone`, `sport`, `tool`, `ml100k`) revealed the following distribution characteristics:

#### A. User Distribution: **100% User Overlap**
* **Train Users = Test Users**: All test users ($100\%$) are present in `train.txt`.
* **Reason**: Collaborative Filtering (LightGCN / MRL4REC) requires learning user embeddings during training for **all** users, so that during evaluation, the model can rank unobserved items for those *same* users.

#### B. User-Level Product Distribution: **0% Overlap (Strictly Disjoint Per User)**
* For any single user $u$:
  $$\text{Items}_{\text{train}}(u) \cap \text{Items}_{\text{test}}(u) = \emptyset$$
* **Average split per user**: $\approx 80\%$ of a user's items go into `train.txt`, and $\approx 20\%$ go into `test.txt` (or $80\% / 10\% / 10\%$ for train/val/test).
* **Reason**: The evaluation tests item recommendation performance on **unseen items** for each user.

#### C. Global Product Distribution: **~99-100% Global Item Overlap**
* Globally across all users, $99-100\%$ of items present in `test.txt` also appear somewhere in `train.txt` (interacted with by *other* users).
* **Reason**: If an item never appeared in `train.txt` for *any* user, its item embedding would remain uninitialized during training.

---

### Benchmark Empirical Summary

| Dataset | Total Users | Train Users | Test Users | Common Users | Avg Train Items/User | Avg Test Items/User | Per-User Item Overlap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`ml100k`** | 943 | 943 | 941 | 941 (100%) | 85.20 | 21.02 | **0.00** |
| **`phone`** | 27,875 | 27,875 | 20,899 | 20,895 (99.9%) | 5.47 | 1.88 | **0.00** |
| **`sport`** | 35,595 | 35,595 | 28,012 | 28,009 (99.9%) | 6.78 | 2.12 | **0.00** |
| **`tool`** | 16,636 | 16,636 | 12,924 | 12,922 (99.9%) | 6.59 | 2.07 | **0.00** |

---

### 3. Comparison with `/Users/lakshgoel/Desktop/BTP/data` & Key Issue

In the current BTP tripartite dataset ([/Users/lakshgoel/Desktop/BTP/data](file:///Users/lakshgoel/Desktop/BTP/data)):

1. **Current BTP Split**:
   - `data/train`: 7,000 users (`account_id` mapped to $0 \dots 6999$)
   - `data/val`: 2,000 users (`account_id` mapped to $0 \dots 1999$) — *Different users!*
   - `data/test`: 1,000 users (`account_id` mapped to $0 \dots 999$) — *Different users!*
   - **Total Unique Users** = 10,000 users ($7,000 + 2,000 + 1,000$).

2. **The Problem with `tripartite_aug_ttv`**:
   The existing `tripartite_aug_ttv` in `MRL4REC/data` used user indices $0 \dots 1999$ from `val` and `test` directly, creating a **user ID mismatch** where User 0 in `train.txt` was a completely different user from User 0 in `valid.txt` / `test.txt`.

---

### 4. How to Correctly Convert BTP `data/` for MRL4REC

To prepare the 10,000 users dataset from BTP for MRL4REC training:

1. **Unify All 10,000 Users**:
   - Read all user-product interactions (`ei_u2pro.npy` or `user_mapping_10k.csv` / `product_mapping_10k.csv`) across `train/`, `val/`, and `test/`.
   - Build a global mapping for all 10,000 users ($0 \dots 9999$) and all unique products ($0 \dots N_{\text{products}}-1$).

2. **Per-User Train/Val/Test Random Split**:
   - For every user $u \in \{0, \dots, 9999\}$, gather all products they interacted with ($I(u)$).
   - Randomly split $I(u)$ into:
     - **Train** ($\approx 80\%$ of items) $\rightarrow$ `train.txt`
     - **Valid** ($\approx 10\%$ of items) $\rightarrow$ `valid.txt`
     - **Test** ($\approx 10\%$ of items) $\rightarrow$ `test.txt`

#### Python Conversion Script Template

Here is a Python script to combine and create a clean MRL4REC dataset:

```python
import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict

random.seed(42)

# 1. Collect all interactions across splits using original string IDs
all_interactions = []
user_set = set()
item_set = set()

for split in ['train', 'val', 'test']:
    u_map = pd.read_csv(f'data/{split}/user_mapping_10k.csv')
    p_map = pd.read_csv(f'data/{split}/product_mapping_10k.csv')
    ei_u2pro = np.load(f'data/{split}/ei_u2pro.npy', allow_pickle=True)
    
    idx2user = dict(zip(u_map['idx'], u_map['account_id']))
    idx2prod = dict(zip(p_map['idx'], p_map['product_id']))
    
    for u_idx, p_idx in ei_u2pro:
        u_id = idx2user[u_idx]
        p_id = idx2prod[p_idx]
        all_interactions.append((u_id, p_id))
        user_set.add(u_id)
        item_set.add(p_id)

# 2. Build global integer index mappings
global_user2idx = {u_id: idx for idx, u_id in enumerate(sorted(user_set))}
global_item2idx = {p_id: idx for idx, p_id in enumerate(sorted(item_set))}

user_items = defaultdict(list)
for u_id, p_id in all_interactions:
    u_idx = global_user2idx[u_id]
    p_idx = global_item2idx[p_id]
    user_items[u_idx].append(p_idx)

# 3. Split 80 / 10 / 10 per user
train_dict, valid_dict, test_dict = defaultdict(list), defaultdict(list), defaultdict(list)

for u_idx, items in user_items.items():
    items = list(set(items)) # deduplicate
    random.shuffle(items)
    n = len(items)
    if n == 1:
        train_dict[u_idx].append(items[0])
    elif n == 2:
        train_dict[u_idx].append(items[0])
        test_dict[u_idx].append(items[1])
    else:
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        train_dict[u_idx] = items[:n_train]
        valid_dict[u_idx] = items[n_train:n_train+n_val]
        test_dict[u_idx] = items[n_train+n_val:]

# 4. Save to MRL4REC data format
out_dir = 'MRL4REC/data/btp_10k'
os.makedirs(out_dir, exist_ok=True)

for split_name, split_dict in [('train.txt', train_dict), ('valid.txt', valid_dict), ('test.txt', test_dict)]:
    with open(os.path.join(out_dir, split_name), 'w') as f:
        for u_idx in sorted(split_dict.keys()):
            if split_dict[u_idx]:
                f.write(f"{u_idx} " + " ".join(map(str, split_dict[u_idx])) + "\n")
```