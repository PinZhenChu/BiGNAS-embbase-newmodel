import logging

import torch
import torch.nn as nn
import wandb
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import numpy as np  
from auxilearn.optim import MetaOptimizer
from dataset import Dataset
from pytorchtools import EarlyStopping
from utils import link_split, load_model

def balance_negative_samples(target_train_link, target_train_label, target_num_items, 
                              num_users, target_item_offset, pos_ratio=1, neg_ratio=4):
    """
    生成負樣本，使得 Label 1 : Label 0 = pos_ratio : neg_ratio
    (局部鎖定亂數版：每次生成的負樣本與順序皆 100% 相同，且具備高速批次運算)
    """
    
    # 計算目標負樣本數量
    pos_count = (target_train_label == 1).sum().item()
    target_neg_count = int(pos_count * neg_ratio / pos_ratio)
    current_neg_count = (target_train_label == 0).sum().item()
    neg_to_add = target_neg_count - current_neg_count
    
    logging.info(f"當前正樣本數: {pos_count}")
    logging.info(f"當前負樣本數: {current_neg_count}")
    logging.info(f"目標負樣本數: {target_neg_count}")
    logging.info(f"需要增加負樣本數: {neg_to_add}")
    
    if neg_to_add <= 0:
        logging.info("已達到目標比例，無需增加負樣本")
        return target_train_link, target_train_label
    
    # 構建所有用戶購買過的 item 集合
    positive_interactions = {}
    item_popularity = {} 

    for user_id, item_id in zip(target_train_link[0].tolist(), target_train_link[1].tolist()):
        if user_id not in positive_interactions:
            positive_interactions[user_id] = set()
        positive_interactions[user_id].add(item_id)
        item_popularity[item_id] = item_popularity.get(item_id, 0) + 1

    # 建立熱門度加權的抽樣機率分佈
    items_list = list(item_popularity.keys())
    counts = np.array(list(item_popularity.values()), dtype=np.float64)
    probs = np.power(counts, 0.5)  
    probs = probs / probs.sum()

    # ==========================================================
    # 🚀 局部鎖定亂數 + 批次效能優化區塊
    # ==========================================================
    # 1. 建立「局部」且「固定」的亂數產生器，不干擾全局環境
    rng = np.random.RandomState(2023) 
    
    negative_pairs_list = []  # 用於保持順序絕對固定
    negative_pairs_set = set() # 用於 O(1) 瞬間檢查重複
    
    # 2. 一次抽足兩倍數量的備用池 (避免 while 迴圈卡死)
    batch_size = neg_to_add * 2 
    sampled_users = rng.randint(0, num_users, size=batch_size)
    sampled_items = rng.choice(items_list, size=batch_size, p=probs)
    
    # 3. 快速篩選
    for u, i in zip(sampled_users, sampled_items):
        if len(negative_pairs_list) >= neg_to_add:
            break
            
        u, i = int(u), int(i)
        if (u, i) not in negative_pairs_set: # 確保這批負樣本內部不重複
            if u not in positive_interactions or i not in positive_interactions[u]:
                negative_pairs_set.add((u, i))
                negative_pairs_list.append((u, i)) # 依序加入 list
                
    # 4. 極端情況下沒抽滿，再單筆補抽
    while len(negative_pairs_list) < neg_to_add:
        u = int(rng.randint(0, num_users))
        i = int(rng.choice(items_list, p=probs))
        if (u, i) not in negative_pairs_set:
            if u not in positive_interactions or i not in positive_interactions[u]:
                negative_pairs_set.add((u, i))
                negative_pairs_list.append((u, i))
                
    negative_pairs = negative_pairs_list
    # ==========================================================
    
    if len(negative_pairs) < neg_to_add:
        logging.warning(f"只生成了 {len(negative_pairs)} 個負樣本，少於目標 {neg_to_add}")
    else:
        logging.info(f"成功生成 {len(negative_pairs)} 個負樣本")
    
    # 將負樣本轉換為 tensor
    if negative_pairs:
        device = target_train_link.device
        
        neg_users = torch.tensor([p[0] for p in negative_pairs], dtype=torch.long, device=device)
        neg_items = torch.tensor([p[1] for p in negative_pairs], dtype=torch.long, device=device)
        neg_link = torch.stack([neg_users, neg_items], dim=0)
        neg_label = torch.zeros(len(negative_pairs), dtype=torch.float, device=device)
        
        # 合併原有的樣本和新生成的負樣本
        new_target_train_link = torch.cat([target_train_link, neg_link], dim=1)
        new_target_train_label = torch.cat([target_train_label, neg_label], dim=0)
        
        # 打印新的標籤分佈
        unique_labels, label_counts = torch.unique(new_target_train_label, return_counts=True)
        logging.info("\n=== 新的標籤分佈 ===")
        for val, cnt in zip(unique_labels.tolist(), label_counts.tolist()):
            logging.info(f"Label {val}: 共有 {cnt} 筆")
        
        ratio = label_counts[0].item() / label_counts[1].item() if len(label_counts) > 1 else float('inf')
        logging.info(f"正負樣本比例 (Label 1 : Label 0): 1 : {ratio:.2f}")
        
        return new_target_train_link, new_target_train_label
    else:
        return target_train_link, target_train_label
    

def meta_optimizeation(
    target_meta_loader,
    replace_optimizer,
    model,
    args,
    criterion,
    replace_scheduler,
    source_edge_index,
    target_train_edge_index,   # ✅ 固定使用 train
):
    device = args.device
    for batch, (target_link, target_label) in enumerate(target_meta_loader):
        if batch < args.descent_step:
            target_link, target_label = target_link.to(device), target_label.to(device)

            replace_optimizer.zero_grad()
            out = model.meta_prediction(
                source_edge_index, target_train_edge_index, target_link
            ).squeeze()
            loss_target = criterion(out, target_label).mean()
            loss_target.backward()
            replace_optimizer.step()
        else:
            break
    replace_scheduler.step()

@torch.no_grad()
def evaluate(name, model, source_edge_index, target_edge_index, link, label):
    model.eval()

    out = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
    try:
        auc = roc_auc_score(label.tolist(), out.tolist())
    except:
        auc = 1.0
    logging.info(f"{name} AUC: {auc:4f}")

    model.train()
    return auc
def get_test_positive_dict(data):
    """
    根據 test link（data.target_test_link）建立 test set user 的正樣本字典。
    回傳: {user_id: [item1, item2, ...]}
    """
    test_user_item_dict = {}
    test_link = data.target_test_link.cpu()
    for u, i in zip(test_link[0], test_link[1]):
        u, i = u.item(), i.item()
        if u not in test_user_item_dict:
            test_user_item_dict[u] = []
        test_user_item_dict[u].append(i)
    return test_user_item_dict

# def evaluate_hit_ratio(
#     model, data, source_edge_index, target_edge_index,
#     top_k,
#     device=None
# ):
#     """
#     ✅ Global HR: 對所有目標物品計算分數，檢查正例是否在 top-k 中
#     （不使用 num_candidates 採樣，而是評估全局排序）
#     """
#     model.eval()

#     # 1. 抓取 Target Domain 全局範圍
#     target_min = data.num_users + data.num_source_items
#     target_max = target_min + data.num_target_items - 1
#     num_target_items = data.num_target_items
    
#     # ✅ 取得 test set 的 user -> positive items 對應關係
#     user_interactions = get_test_positive_dict(data)
#     sim_users = list(user_interactions.keys())
#     print(f"✅ Test set user count: {len(sim_users)}")

#     source_edge_index = source_edge_index.to(device)
#     target_edge_index = target_edge_index.to(device)

#     total_users = 0
#     hit_count = 0
    
#     # 2. 預先建立全局的 Item Tensor (包含所有的 Target Items)
#     global_item_tensor = torch.arange(target_min, target_max + 1, dtype=torch.long, device=device)

#     with torch.no_grad():
#         for user_id in sim_users:
#             pos_items = user_interactions.get(user_id, set())
#             if len(pos_items) == 0:
#                 continue
            
#             if len(pos_items) > 1:
#                 print(f"⚠️ Warning: User {user_id} has {len(pos_items)} positives in test set.")

#             # ✅ 只取第一個正樣本
#             pos_item = list(pos_items)[0]

#             # ✅ 建立對應此 User 的 Tensor (大小與 target item 總數相同)
#             user_tensor = torch.full((num_target_items,), user_id, dtype=torch.long, device=device)
#             link = torch.stack([user_tensor, global_item_tensor], dim=0)

#             # 計算該 User 對 "所有商品" 的預測分數
#             scores = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
            
#             # 3. 使用 GPU 原生的 topk 加速排序
#             top_scores, top_indices = torch.topk(scores, top_k)
            
#             # top_indices 是從 0 開始的相對位置，必須加上 target_min 轉回真實的 Item ID
#             top_k_items = (target_min + top_indices).tolist()

#             # 4. 檢查正樣本是否在 top-k 中
#             if pos_item in top_k_items:
#                 hit_count += 1
            
#             total_users += 1

#     hit_ratio = hit_count / total_users if total_users > 0 else 0.0
#     logging.info(f"[Global HR@{top_k}] Users={total_users}, Hits={hit_count}, Hit Ratio={hit_ratio:.4f}")
#     return hit_ratio
def evaluate_hit_ratio(
    model, data, source_edge_index, target_edge_index,
    top_k,
    device=None
):
    """
    ✅ Global HR: 包含 Rank-based 計算、全用戶 Rank 列印與分佈統計
    """
    model.eval()

    # 1. 抓取 Target Domain 全局範圍
    target_min = data.num_users + data.num_source_items
    target_max = target_min + data.num_target_items - 1
    num_target_items = data.num_target_items
    
    user_interactions = get_test_positive_dict(data)
    sim_users = list(user_interactions.keys())
    
    source_edge_index = source_edge_index.to(device)
    target_edge_index = target_edge_index.to(device)

    total_users = 0
    hit_count = 0
    
    # ✅ 新增：初始化分佈計數器 (僅在第一輪 top_k 使用)
    dist_counts = {
        "Top-10": 0,
        "11-50": 0,
        "51-100": 0,
        "101-200": 0,
        "201-500": 0,
        "501-1000": 0,
        "1001+": 0
    }

    # 預先建立全局的 Item Tensor
    global_item_tensor = torch.arange(target_min, target_max + 1, dtype=torch.long, device=device)

    with torch.no_grad():
        for user_id in sim_users:
            pos_items = user_interactions.get(user_id, set())
            if len(pos_items) == 0:
                continue
            
            # 取得唯一正樣本
            pos_item = list(pos_items)[0]

            # 建立 Link Tensor 計算該 User 對 "所有 Target 商品" 的分數
            user_tensor = torch.full((num_target_items,), user_id, dtype=torch.long, device=device)
            link = torch.stack([user_tensor, global_item_tensor], dim=0)

            # 模型預測
            scores = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
            
            # --- 計算 Rank ---
            pos_item_idx = pos_item - target_min 
            pos_score = scores[pos_item_idx]
            rank = (scores > pos_score).sum().item() + 1
            
            # ✅ 新增：統計與詳細列印邏輯 (只在跑第一輪評估時執行)
            if top_k == 10:
                # 累加分佈統計
                if rank <= 10: dist_counts["Top-10"] += 1
                elif rank <= 50: dist_counts["11-50"] += 1
                elif rank <= 100: dist_counts["51-100"] += 1
                elif rank <= 200: dist_counts["101-200"] += 1
                elif rank <= 500: dist_counts["201-500"] += 1
                elif rank <= 1000: dist_counts["501-1000"] += 1
                else: dist_counts["1001+"] += 1

                # 根據排名列印不同符號
                if rank <= 500:
                    print(f"🔥 [Debug] User {user_id} | Rank: {rank}/{num_target_items} (Score: {pos_score:.4f})")
                else:
                    print(f"⚪ [Detail] User {user_id} | Rank: {rank}/{num_target_items}")

            # 檢查排名是否在當前 Top-K 內
            if rank <= top_k:
                hit_count += 1
            
            total_users += 1

    # ✅ 新增：在第一輪結束後印出統計小報表
    if top_k == 10:
        print("\n" + "="*50)
        print("📊 RANK DISTRIBUTION SUMMARY")
        print("-" * 50)
        for range_name, count in dist_counts.items():
            percentage = (count / total_users) * 100 if total_users > 0 else 0
            bar = "■" * (int(percentage / 2)) # 每 2% 顯示一個方塊
            print(f"{range_name:10} : {count:4} users ({percentage:5.2f}%) {bar}")
        print("="*50 + "\n")

    hit_ratio = hit_count / total_users if total_users > 0 else 0.0
    logging.info(f"[Global HR@{top_k}] Users={total_users}, Hits={hit_count}, Hit Ratio={hit_ratio:.4f}")
    return hit_ratio
# 🔍 統計每個 cold item 在 test set 中出現的次數（有幾個 user 買過）
def count_cold_item_occurrences(data, cold_item_set):
    item_count = {item: 0 for item in cold_item_set}
    test_link = data.target_test_link.cpu().numpy()
    for u, i in zip(*test_link):
        if i in cold_item_set:
            item_count[i] += 1
    return item_count

def evaluate_er_hit_ratio(
    model, data, source_edge_index, target_edge_index,
    cold_item_set,
    top_k, 
    device=None
):
    model.eval()

    # 1. 抓取 Target Domain 全局範圍
    target_min = data.num_users + data.num_source_items
    target_max = target_min + data.num_target_items - 1
    num_target_items = data.num_target_items
    
    user_interactions = get_test_positive_dict(data)
    sim_users = list(user_interactions.keys())

    source_edge_index = source_edge_index.to(device)
    target_edge_index = target_edge_index.to(device)

    total_users = 0
    cold_item_hit_count = 0
    cold_item_ranks = []  # 儲存 cold item 被排進去時的排名

    # 2. 預先建立全局的 Item Tensor (包含所有的 Target Items)
    global_item_tensor = torch.arange(target_min, target_max + 1, dtype=torch.long, device=device)

    with torch.no_grad():
        for user_id in sim_users:
            # 建立對應此 User 的 Tensor (大小與 target item 總數相同)
            user_tensor = torch.full((num_target_items,), user_id, dtype=torch.long, device=device)
            link = torch.stack([user_tensor, global_item_tensor], dim=0)

            # 計算該 User 對 "所有商品" 的預測分數
            scores = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
            
            # 3. 使用 GPU 原生的 topk 加速排序，避免轉回 CPU 用 list.sort()
            top_scores, top_indices = torch.topk(scores, top_k)
            
            # top_indices 是從 0 開始的相對位置，必須加上 target_min 轉回真實的 Item ID
            top_k_items = (target_min + top_indices).tolist()

            # 4. 統計冷啟動商品是否成功擠進全局 Top-K 名單中
            cold_hits = [item for item in top_k_items if item in cold_item_set]
            if cold_hits:
                cold_item_hit_count += 1
                for cold_item in cold_hits:
                    rank = top_k_items.index(cold_item) + 1  # 1-based rank
                    cold_item_ranks.append(rank)

            total_users += 1

    er_ratio = cold_item_hit_count / total_users if total_users > 0 else 0.0
    
    # 印出時標註是 Global ER
    logging.info(f"[Global ER@{top_k}] Users={total_users}, Cold Item Hits={cold_item_hit_count}, ER Ratio={er_ratio:.4f}")

    return er_ratio

def evaluate_multiple_topk(model, data, source_edge_index, target_edge_index, cold_item_set, device):
    topk_list = [10, 15, 20, 25, 30]
    print("\n📊 Evaluation for multiple top-K values:")
    print("="*80)
    for k in topk_list:
        
        # ✅ Hit Ratio (Global)
        hr = evaluate_hit_ratio(
            model=model,
            data=data,
            source_edge_index=source_edge_index,
            target_edge_index=target_edge_index,
            top_k=k,
            device=device
        )

        # ✅ ER (Cold Item Hit Ratio)
        er = evaluate_er_hit_ratio(
            model=model,
            data=data,
            source_edge_index=source_edge_index,
            target_edge_index=target_edge_index,
            cold_item_set=cold_item_set,
            top_k=k,
            device=device
        )
        

    # for k in [150, 200, 300, 400, 500, 1000]:

    #     # ER 計算改為 Global (不再傳入 num_candidates)
    #     er = evaluate_er_hit_ratio(
    #         model=model,
    #         data=data,
    #         source_edge_index=source_edge_index,
    #         target_edge_index=target_edge_index,
    #         cold_item_set=cold_item_set,
    #         top_k=k,
    #         device=device
    #     )

# === 保留這個（原本完整版本，能分辨 target / source）===
def check_all_edges(edge_index, num_users, num_source_items, num_target_items, name, is_target=False):
    if edge_index is None or edge_index.numel() == 0:
        logging.info(f"[{name}] empty (skip check)")
        return

    u, v = edge_index
    u_min, u_max = u.min().item(), u.max().item()
    v_min, v_max = v.min().item(), v.max().item()

    if is_target:
        valid_min = num_users + num_source_items
        valid_max = num_users + num_source_items + num_target_items - 1
    else:
        valid_min = num_users
        valid_max = num_users + num_source_items - 1

    logging.info(
        f"[{name}] users {u_min}~{u_max} (limit {num_users-1}), "
        f"items {v_min}~{v_max}, valid [{valid_min}~{valid_max}]"
    )

    if u_min < 0 or u_max >= num_users:
        raise ValueError(f"[{name}] user id 越界: {u_min} ~ {u_max}, 應該在 [0, {num_users-1}]")
    if v_min < valid_min or v_max > valid_max:
        raise ValueError(f"[{name}] item id 越界: {v_min} ~ {v_max}, 應該在 [{valid_min}, {valid_max}]")

# === 新增一個給「全局 num_nodes」用的 ===
def check_all_edges_global(edge_index, num_nodes, name):
    if edge_index is None or edge_index.numel() == 0:
        logging.info(f"[{name}] empty (skip check)")
        return
    if edge_index.min().item() < 0 or edge_index.max().item() >= num_nodes:
        raise ValueError(
            f"❌ {name}: index 越界 (min={edge_index.min().item()}, max={edge_index.max().item()}, num_nodes={num_nodes})"
        )
    u, v = edge_index
    logging.info(
        f"[{name}] OK: u[{u.min().item()}~{u.max().item()}], v[{v.min().item()}~{v.max().item()}], num_nodes={num_nodes}"
    )

def train(model, perceptor, data, args, split_result, summary=None):
    """
    ✅ 注：此函数开始时假设所有输入已从新初始化，不依赖任何全局状态
    """
    device = args.device
    
    # 🛡️ 重要：在訓練開始前清理所有可能的殘留狀態
    logging.info("[Train] 開始清理殘留資源...")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # 🔄 確保模型處於訓練模式（有時前面的evaluate會改變這個狀態）
    model.train()
    perceptor.train()
    
    data = data.to(device)
    model = model.to(device)
    perceptor = perceptor.to(device)
    
    logging.info("[Train] 資源清理和初始化完成")

    # ✅ edge_index 只用 split_result (global id)
    source_edge_index       = split_result["source_train_edge_index"].to(device)
    target_train_edge_index = split_result["target_train_edge_index"].to(device)
    target_valid_edge_index = split_result["target_valid_edge_index"].to(device)
    target_test_edge_index  = split_result["target_test_edge_index"].to(device)

    # ✅ link & label 還是從 link_split 拿，但 edge_index 忽略
    (
        _,
        source_label,
        source_link,
        _,
        target_train_label,
        target_train_link_from_split,
        target_valid_link,
        target_valid_label,
        target_test_link,
        target_test_label,
        _,
    ) = link_split(data)

    # ✅ 全局節點數
    num_nodes = args.num_users + args.num_source_items + args.num_target_items
    check_all_edges_global(source_edge_index, num_nodes, "source_edge_index")
    check_all_edges_global(target_train_edge_index, num_nodes, "target_train_edge_index")
    check_all_edges_global(target_valid_edge_index, num_nodes, "target_valid_edge_index")
    check_all_edges_global(target_test_edge_index, num_nodes, "target_test_edge_index")


    data.target_test_link = target_test_link
    # =====================================================================
    # ============== 🚀 終極修正版：精準對齊標籤與連線長度 ==============
    # =====================================================================
    # 1. 取得加減邊後的完整連線 (這已經包含 search.py 合併的所有假邊)
    target_train_link = target_train_edge_index.detach().clone()
    
    # 2. 🛡️ 關鍵：初始化 new_labels，長度必須等於「目前的連線總數」
    total_link_count = target_train_link.shape[1]
    new_labels = torch.ones(total_link_count, device=device) # 預設全部先給正標籤 1.0

    # 3. 繼承原始負樣本標籤
    # 找出原始 split 產出的標籤中哪些是 0.0，並對應回去
    # 注意：target_train_label 是由 link_split 產出的原始標籤
    orig_neg_mask = (target_train_label == 0.0)
    # 只對原始長度範圍內的標籤進行 0.0 覆蓋
    new_labels[:len(target_train_label)][orig_neg_mask] = 0.0

    # 4. 處理「被刪除的邊」(Suppressed)：將原本為 1.0 的改為 0.0
    if summary is not None and summary["E_remove_suppress"].numel() > 0:
        remove_edges = summary["E_remove_suppress"].t().tolist()
        remove_set = set(tuple(e) for e in remove_edges)
        
        # 取得 link_split 產出的原始邊備份
        original_links = target_train_link_from_split.t().tolist() 
        for idx, edge in enumerate(original_links):
            if tuple(edge) in remove_set:
                if idx < len(new_labels): # 安全檢查
                    new_labels[idx] = 0.0  # 強制降級為負樣本

    # 5. 正式更新標籤變數
    target_train_label = new_labels
    data.target_train_edge_index = target_train_edge_index
    # =====================================================================
    # ✅ 執行負樣本平衡 (補足到目標 1:2 比例，此時會產出 80680 筆資料)
    target_item_offset = args.num_users + args.num_source_items
    target_train_link, target_train_label = balance_negative_samples(
        target_train_link=target_train_link,
        target_train_label=target_train_label,
        target_num_items=args.num_target_items,
        num_users=args.num_users,
        target_item_offset=target_item_offset,
        pos_ratio=1,
        neg_ratio=0
    )

     # 🛡️ 確保在這之後才建立 Dataset 物件，且長度已經是最終平衡後的結果[cite: 3]
    target_train_set = Dataset(
        target_train_link.detach().to("cpu"), # 增加 detach() 確保安全
        target_train_label.detach().to("cpu"),
    )
    
    
    source_set_size = source_link.shape[1]
    train_set_size = target_train_link.shape[1]
    val_set_size = target_valid_link.shape[1]
    test_set_size = target_test_link.shape[1]
    logging.info(f"Train set size: {train_set_size}")
    logging.info(f"Valid set size: {val_set_size}")
    logging.info(f"Test set size: {test_set_size}")

    target_train_set = Dataset(
        target_train_link.to("cpu"),
        target_train_label.to("cpu"),
    )
    target_train_loader = DataLoader(
        target_train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # ✅ 改為0防止多進程污染
        collate_fn=target_train_set.collate_fn,
    )

    source_batch_size = int(args.batch_size * train_set_size / source_set_size)
    source_train_set = Dataset(source_link.to("cpu"), source_label.to("cpu"))
    source_train_loader = DataLoader(
        source_train_set,
        batch_size=source_batch_size,
        shuffle=True,
        num_workers=0,  # ✅ 改為0防止多進程污染
        collate_fn=source_train_set.collate_fn,
    )

    target_meta_loader = DataLoader(
        target_train_set,
        batch_size=args.meta_batch_size,
        shuffle=True,
        num_workers=0,  # ✅ 改為0防止多進程污染
        collate_fn=target_train_set.collate_fn,
    )
    target_meta_iter = iter(target_meta_loader)
    source_meta_batch_size = int(
        args.meta_batch_size * train_set_size / source_set_size
    )
    source_meta_loader = DataLoader(
        source_train_set,
        batch_size=source_meta_batch_size,
        shuffle=True,
        num_workers=0,  # ✅ 改為0防止多進程污染
        collate_fn=source_train_set.collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    perceptor_optimizer = torch.optim.Adam(
        perceptor.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    meta_optimizer = MetaOptimizer(
        meta_optimizer=perceptor_optimizer,
        hpo_lr=args.hpo_lr,
        truncate_iter=3,
        max_grad_norm=10,
    )

    model_param = [
        param for name, param in model.named_parameters() if "preds" not in name
    ]
    replace_param = [
        param for name, param in model.named_parameters() if name.startswith("replace")
    ]
    replace_optimizer = torch.optim.Adam(replace_param, lr=args.lr)
    replace_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        replace_optimizer, T_max=args.T_max
    )

    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        path=args.model_path,
        trace_func=logging.info,
    )

    criterion = nn.BCELoss(reduction="none")
    iteration = 0
    for epoch in range(args.epochs):
        for (source_link, source_label), (target_link, target_label) in zip(
            source_train_loader, target_train_loader
        ):
            torch.cuda.empty_cache()
            source_link = source_link.to(device)
            source_label = source_label.to(device)
            target_link = target_link.to(device)
            target_label = target_label.to(device)
            weight_source = perceptor(source_link[1], source_edge_index, model)

            optimizer.zero_grad()
            source_out = model(
                source_edge_index, target_train_edge_index, source_link, is_source=True
            ).squeeze()
            target_out = model(
                source_edge_index, target_train_edge_index, target_link, is_source=False
            ).squeeze()
            source_loss = (
                criterion(source_out, source_label).reshape(-1, 1) * weight_source
            ).sum()
            target_loss = criterion(target_out, target_label).mean()
            loss = source_loss + target_loss if args.use_meta else target_loss
            loss.backward()
            optimizer.step()

            iteration += 1
            if (
                args.use_source
                and args.use_meta
                and iteration % args.meta_interval == 0
            ):
                logging.info(f"Entering meta optimization, iteration: {iteration}")
                meta_optimizeation(
                    target_meta_loader,
                    replace_optimizer,
                    model,
                    args,
                    criterion,
                    replace_scheduler,
                    source_edge_index,
                    target_train_edge_index,
                )

                try:
                    target_meta_link, target_meta_label = next(target_meta_iter)
                except StopIteration:
                    target_meta_iter = iter(target_meta_loader)
                    target_meta_link, target_meta_label = next(target_meta_iter)

                target_meta_link, target_meta_label = (
                    target_meta_link.to(device),
                    target_meta_label.to(device),
                )
                optimizer.zero_grad()
                target_out = model(
                    source_edge_index,
                    target_train_edge_index,
                    target_meta_link,
                    is_source=False,
                ).squeeze()
                meta_loss = criterion(target_out, target_meta_label).mean()

                for (source_link, source_label), (target_link, target_label) in zip(
                    source_meta_loader, target_meta_loader
                ):
                    source_link, source_label = source_link.to(device), source_label.to(
                        device
                    )
                    target_link, target_label = target_link.to(device), target_label.to(
                        device
                    )
                    weight_source = perceptor(source_link[1], source_edge_index, model)

                    optimizer.zero_grad()
                    source_out = model(
                        source_edge_index,
                        target_train_edge_index,
                        source_link,
                        is_source=True,
                    ).squeeze()
                    target_out = model(
                        source_edge_index,
                        target_train_edge_index,
                        target_link,
                        is_source=False,
                    ).squeeze()
                    source_loss = (
                        criterion(source_out, source_label).reshape(-1, 1)
                        * weight_source
                    ).sum()
                    target_loss = criterion(target_out, target_label).mean()
                    meta_train_loss = (
                        source_loss + target_loss if args.use_meta else target_loss
                    )
                    break

                torch.cuda.empty_cache()
                meta_optimizer.step(
                    train_loss=meta_train_loss,
                    val_loss=meta_loss,
                    aux_params=list(perceptor.parameters()),
                    parameters=model_param,
                    return_grads=True,
                    entropy=None,
                )
        train_auc = evaluate(
            "Train",
            model,
            source_edge_index,
            target_train_edge_index,
            target_train_link,
            target_train_label,
        )
        val_auc = evaluate(
            "Valid",
            model,
            source_edge_index,
            target_train_edge_index,
            target_valid_link,
            target_valid_label,
        )

        logging.info(
            f"[Epoch: {epoch}]Train Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, Valid AUC: {val_auc:.4f}"
        )
        wandb.log(
            {
                "loss": loss,
                "train_auc": train_auc,
                "val_auc": val_auc
            },
            step=epoch,
        )

        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

        lr_scheduler.step()

    # model = load_model(args).to(device)
    # ✅ 改成這兩行：(從 EarlyStopping 存檔的路徑讀取最佳權重)
    logging.info(f"載入最佳模型權重: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    # ========================================================
    # 🔍 [診斷區間] 檢查 A、B、C：確認資料結構與分佈
    # ========================================================
    print("\n" + "="*60)
    print("=== [檢查 A] 節點 ID 空間分佈 (確認是否有重疊) ===")
    user_min, user_max = source_link[0].min().item(), source_link[0].max().item()
    src_min, src_max = source_link[1].min().item(), source_link[1].max().item()
    tgt_min, tgt_max = target_train_link[1].min().item(), target_train_link[1].max().item()

    print(f"User ID 範圍: {user_min} ~ {user_max}")
    print(f"Source Item ID 範圍: {src_min} ~ {src_max}")
    print(f"Target Item ID 範圍: {tgt_min} ~ {tgt_max}")

    if src_min <= tgt_max and tgt_min <= src_max:
        print("🚨 警告：Source 和 Target 的 Item ID 空間確實發生了重疊！")
    else:
        print("✅ 正常：Source 和 Target 的 Item ID 空間是獨立的。")

    print("\n=== [檢查 B] Target Domain 真實熱門排行榜 ===")
    items, counts = torch.unique(target_train_link[1], return_counts=True)
    sorted_counts, sorted_indices = torch.sort(counts, descending=True)
    top_popular_items = items[sorted_indices][:10]

    for i in range(min(10, len(top_popular_items))):
        item_id = top_popular_items[i].item()
        count = sorted_counts[i].item()
        print(f"Item ID: {item_id:<6} | 訓練集被購買次數: {count}")

    print("\n=== [檢查 C] Target Train Label 分佈 ===")
    unique_labels, label_counts = torch.unique(target_train_label, return_counts=True)
    for val, cnt in zip(unique_labels.tolist(), label_counts.tolist()):
        print(f"Label {val}: 共有 {cnt} 筆")
    print("💡 如果沒有 Label 0，代表模型缺乏'未購買'的負樣本學習！")
    print("="*60 + "\n")
    # ========================================================

    # 執行 Top-K 印出與變異數檢查
    print_top_target_items(
        model=model, 
        source_edge_index=source_edge_index, 
        target_train_edge_index=target_train_edge_index, 
        target_test_link=target_test_link, 
        target_test_label=target_test_label, 
        args=args
    )
    # ✅ Global HR 評估 (已改為對所有物品計算，無需 num_candidates)
    # evaluate_hit_ratio(
    #     model=model,
    #     data=data,
    #     source_edge_index=source_edge_index,
    #     target_edge_index=target_train_edge_index,
    #     top_k=args.top_k,
    #     device=device,
    # )
    # cold_item_id = find_cold_item_strict(data, target_train_edge_index, target_test_edge_index)
    if args.cold_item_id >= 0:
        cold_item_id = args.cold_item_id
        target_min = args.num_users + args.num_source_items
        target_max = args.num_users + args.num_source_items + args.num_target_items - 1
        assert target_min <= cold_item_id <= target_max, \
            f"❌ cold_item_id={cold_item_id} 不在 target 範圍 [{target_min}, {target_max}]"
        logging.info(f"[ColdItem] 使用指定 cold_item_id={cold_item_id}")
    else:
        # 保險起見，還是留一個 fallback（自動選）
        cold_item_id = None
        logging.info("[ColdItem] 未指定，跳過 ER 評估")

    # if cold_item_id is not None:
    #     evaluate_er_hit_ratio(
    #         model=model,
    #         data=data,
    #         source_edge_index=source_edge_index,
    #         target_edge_index=target_train_edge_index,
    #         cold_item_set={cold_item_id},
    #         top_k=args.top_k,
    #         num_candidates=99,
    #         device=device,
    #     )


    # logging.info(f"Hit Ratio (no injection): {pre_hit_ratio:.4f}")
    test_auc = evaluate(
        "Test",
        model,
        source_edge_index,
        target_train_edge_index,
        target_test_link,
        target_test_label,
    )
    logging.info(f"Test AUC: {test_auc:.4f}")
    wandb.log({"Test AUC": test_auc})
    evaluate_multiple_topk(
        model=model,
        data=data,
        source_edge_index=source_edge_index,
        target_edge_index=target_train_edge_index,
        cold_item_set={cold_item_id},   # 注意這邊是 set，不是 cold_item_id=
        device=device
    )
    
        # === 存下 source_item_embedding ===
    # source_emb = model.source_item_embedding.weight.detach().cpu().numpy()
    # np.save("source_item_embedding.npy", source_emb)
    # np.savetxt("source_item_embedding.csv", source_emb, delimiter=",")
    # logging.info(f"✅ Saved source_item_embedding: shape={source_emb.shape}")



def print_top_target_items(model, source_edge_index, target_train_edge_index, target_test_link, target_test_label, args):
    """
    印出 Target Domain 分數最高的前 10 個 item，並精準標記 Train(已購買) 與 HIT(測試集命中)
    """
    device = args.device
    model.eval()
    with torch.no_grad():
        # 1. 動態抓取 Target Item ID 的真實範圍
        target_item_min = target_train_edge_index[1].min().item()
        target_item_max = target_train_edge_index[1].max().item()
        num_target_candidates = target_item_max - target_item_min + 1
        k = min(10, num_target_candidates)
        
        # ========================================================
        # 🔍 提前做 [檢查 D]：預測分數多樣性 
        # ========================================================
        user_links_0 = torch.full((2, num_target_candidates), 0, dtype=torch.long, device=device)
        user_links_0[1, :] = torch.arange(target_item_min, target_item_max + 1, device=device)
        scores_0 = model(source_edge_index, target_train_edge_index, user_links_0, is_source=False).squeeze()
        
        print("\n" + "="*60)
        print("=== [檢查 D] 預測分數多樣性 (Model Collapse 檢查) ===")
        score_var = torch.var(scores_0).item()
        print(f"User 0 對 Target Domain 所有物品預測分數的變異數: {score_var:.8f}")
        print(f"預測分數最高分: {scores_0.max().item():.4f}, 最低分: {scores_0.min().item():.4f}")
        print("💡 如果變異數極小 (例如 0.00000x)，代表發生 Model Collapse！")
        print("="*60 + "\n")
        
        # 2. 建立 Target Domain 的 Ground Truth (區分 Train 和 Test)
        train_interactions = {}
        for u, i in zip(target_train_edge_index[0].cpu().numpy(), target_train_edge_index[1].cpu().numpy()):
            train_interactions.setdefault(u, set()).add(i)
            
        test_interactions = {}
        pos_test_mask = target_test_label == 1
        pos_test_links = target_test_link[:, pos_test_mask]
        for u, i in zip(pos_test_links[0].cpu().numpy(), pos_test_links[1].cpu().numpy()):
            test_interactions.setdefault(u, set()).add(i)

        # --- 印出格式化的表頭 ---
        col_width = 8  # 設定固定欄寬
        header_cols = [f"{'user id':<7}"] + [f"top{i+1} id(score)".ljust(col_width) for i in range(k)]
        print(" | ".join(header_cols))
        
        for user_id in range(50):  # 先印前 5 個 user 來觀察
            user_links = torch.full((2, num_target_candidates), user_id, dtype=torch.long, device=device)
            user_links[1, :] = torch.arange(target_item_min, target_item_max + 1, device=device)
            
            scores = model(source_edge_index, target_train_edge_index, user_links, is_source=False).squeeze()
            top_scores, top_indices = torch.topk(scores, k)
            
            user_train_items = train_interactions.get(user_id, set())
            user_test_items = test_interactions.get(user_id, set())
            
            row_data = [f"{user_id:<7}"] 
            for i in range(len(top_indices)):
                item_id = target_item_min + top_indices[i].item()
                score = top_scores[i].item()
                
                # 標記邏輯
                if item_id in user_test_items:
                    item_str = f"{item_id}*HIT*"
                elif item_id in user_train_items:
                    item_str = f"{item_id}(Train)"
                else:
                    item_str = f"{item_id}"
                    
                # 組合字串並使用 ljust 填充空格對齊
                cell_str = f"{item_str}({score:.4f})"
                row_data.append(cell_str.ljust(col_width))
            
            print(" | ".join(row_data))
            
    model.train()