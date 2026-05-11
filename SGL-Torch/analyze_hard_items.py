import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

def find_hard_items_and_export_verbose(
    model,
    groupA_ids,
    hard_user_ids,
    num_users,
    num_source_items,
    num_target_items,
    k_source,
    save_dir,
    preview_top_users
):
    """
    🔥 最純粹版本：
        ✔ Source domain：B 喜歡但 Hard User 不喜歡 (mean_B - mean_H)
        ❌ Target domain：全部移除
    """

    import os
    import torch
    import numpy as np
    import torch.nn.functional as F
    import pandas as pd

    os.makedirs(save_dir, exist_ok=True)
    device = model.device
    model.lightgcn.eval()

    # === Step 1. 匯出 embedding ===
    with torch.no_grad():
        uemb, iemb = model.lightgcn._forward_gcn(model.lightgcn.norm_adj)
        uemb = F.normalize(uemb, dim=1)
        iemb = F.normalize(iemb, dim=1)
    print("=" * 80)
    print(f"[1] Embedding ready: user={tuple(uemb.shape)}, item={tuple(iemb.shape)}")

    total_items = iemb.size(0)
    if num_source_items == 0 and num_target_items == 0:
        num_source_items = total_items // 2
        num_target_items = total_items - num_source_items
        print(f"⚠️ 自動推測 item 範圍: source={num_source_items}, target={num_target_items}")

    B_users = torch.tensor(groupA_ids, dtype=torch.long, device=device)
    hardU = torch.tensor(hard_user_ids, dtype=torch.long, device=device)

    # === Step 2. 計算偏好差距 Δ = mean_B - mean_H ===
    with torch.no_grad():
        scores_B = model.lightgcn.predict(B_users)
        mean_B = scores_B.mean(dim=0, keepdim=True)

        scores_H = model.lightgcn.predict(hardU)
        mean_H = scores_H.mean(dim=0, keepdim=True)

        delta = (mean_B - mean_H).squeeze(0)
        delta = torch.nan_to_num(delta, nan=0.0)

    print("=" * 80)
    print("[2] 已計算 Δ = mean_B - mean_H")

    # === 只取 Source domain 的 Δ ===
    delta_src = delta[:num_source_items]
    vals_s, idx_s = torch.topk(delta_src, k=min(k_source, delta_src.numel()))

    selected_source_global_ids = [num_users + i for i in idx_s.cpu().tolist()]
    print("=" * 80)
    print(f"[Source] Δ 最大的 {k_source} 個 source items:")
    print(selected_source_global_ids)

    # === Step 3. 加邊 ===
    all_source_edges = []
    preview_log = []

    print("\n=== All added source edges ===")
    for uid in hard_user_ids:
        for iid_global in selected_source_global_ids:
            all_source_edges.append((uid, iid_global))

            # ⭐⭐⭐ 新增：列印所有加上的邊 ⭐⭐⭐
            print(f"  + user {uid}  ->  item {iid_global}")

            if len(preview_log) < preview_top_users * k_source:
                local_src_idx = iid_global - num_users
                preview_log.append((
                    uid,
                    iid_global,
                    mean_H[0, local_src_idx].item(),
                    mean_B[0, local_src_idx].item(),
                    delta_src[local_src_idx].item()
                ))

    # === Step 4. 輸出為 tensor / csv ===
    def make_edge_tensor(edge_list):
        if len(edge_list) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edge_list, dtype=torch.long).t()

    E_add_source = make_edge_tensor(all_source_edges)
    np.save(os.path.join(save_dir, "E_add_source.npy"), E_add_source.cpu().numpy())

    print("=" * 80)
    print(f"[3] 完成：Hard Users = {len(hard_user_ids)}")
    print(f"    Source 假邊數量：{E_add_source.size(1)} 條")

    # === 預覽前幾筆 ===
    print("=" * 80)
    print(f"[4] 🔍 Hard User 加邊預覽 (前 {preview_top_users} 位)")
    print(f"{'User':>6} | {'Item':>6} | {'HardScore':>10} | {'B_Mean':>10} | {'Δ':>10}")
    print("-" * 60)
    for uid, iid, sc_h, sc_b, diff in preview_log:
        print(f"{uid:>6d} | {iid:>6d} | {sc_h:>10.6f} | {sc_b:>10.6f} | {diff:>10.6f}")

    # === CSV ===
    src_df = pd.DataFrame(E_add_source.cpu().numpy().T, columns=["user_id", "item_id"])
    src_df.to_csv(os.path.join(save_dir, "E_add_source.csv"), index=False)

    print("=" * 80)
    print("[5] 輸出完成：source 假邊 .npy + CSV 版")

    return E_add_source, src_df
