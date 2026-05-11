import os
import math
import logging
import numpy as np
import torch
import torch.nn.functional as F


def _tensor2set(edge_index: torch.Tensor):
    """
    將 edge_index (shape = [2, E]) 轉成 Python 的 set[(u, i), ...]。

    為什麼要這樣做？
    - 方便用 O(1) in / remove / add 來檢查邊是否存在
    - 適合做「加邊 / 減邊」的集合運算（避免重複邊）

    參數：
        edge_index: torch.LongTensor, shape=[2, num_edges]
            第 0 列：user id
            第 1 列：item id
            都是在「global id 空間」裡。

    回傳：
        s: set[tuple[int, int]]
            每個元素是 (user_id, item_id)
    """
    # edge_index.t() -> shape=[E, 2]，每列為 [u, i]
    # .tolist() -> List[List[int, int]]
    # map(tuple, ...) -> List[(u, i)]
    # set(...) -> set of edges
    return set(map(tuple, edge_index.t().tolist()))


def _apply_add(edge_index: torch.Tensor, additions):
    """
    在原本的 edge_index 上「加入」一些 (u, i) 邊，並回傳新的 edge_index。

    注意：
    - 使用 set 來去重：即使 additions 中有重複邊，最後仍只保留一條。
    - 所有 id 都是在 global id 空間裡，不做 offset。

    參數：
        edge_index: torch.LongTensor, shape=[2, E]
            原始邊集合（global id）
        additions: Iterable[tuple[int, int]]
            要加入的邊 (u, i)

    回傳：
        new_edge_index: torch.LongTensor, shape=[2, E_new]
            加完邊後的完整邊集合
    """
    s = _tensor2set(edge_index)  # 原本邊集合

    # 把要加的邊都塞進 set（自動處理重複）
    for u, i in additions:
        s.add((int(u), int(i)))

    # set -> list -> tensor，再轉回 shape=[2, E_new]
    return torch.tensor(list(s), dtype=torch.long).t()


def _apply_remove(edge_index: torch.Tensor, removals):
    """
    在原本的 edge_index 上「移除」一些 (u, i) 邊，並回傳新的 edge_index。

    參數：
        edge_index: torch.LongTensor, shape=[2, E]
            原始邊集合（global id）
        removals: Iterable[tuple[int, int]]
            要刪掉的邊 (u, i)

    回傳：
        new_edge_index: torch.LongTensor, shape=[2, E_new]
            減完邊後的完整邊集合
    """
    s = _tensor2set(edge_index)  # 原本邊集合

    for u, i in removals:
        key = (int(u), int(i))
        if key in s:  # 只有真的存在的邊才移除，避免 KeyError
            s.remove(key)

    return torch.tensor(list(s), dtype=torch.long).t()


class HardUserInjector:
    """
    ⭐ 乾淨版 HardUserInjector（全 Hard User 都執行加邊 & 減邊，沒有隨機比例）

    整體流程邏輯：

    1. 選出「冷門商品」 cold_item，並找出：
       - GroupA：有買過 cold_item 的 user
       - GroupB：沒買過 cold_item 的 user

    2. 從 GroupB 中選出「Hard Users」
       - 定義：對 GroupA 的使用者 embedding 最不相似的那一群
       - 方法：cos similarity -> dist = 1 - max_sim -> 按 dist 由大到小取 top_ratio%

    3. 對「所有 Hard Users」做兩件事（沒有任何比例 / 隨機）：
       (a) 加邊（promote 冷門商品）：
           - 每個 Hard User 都加一條 (user, cold_item_global) 邊

       (b) 減邊（suppress popular items）：
           - 先用 target_train_edge_index 找出 target domain 的熱門商品 popular_items
           - 對所有 Hard User × popular_items 的「原本存在」邊，全部刪掉

    4. 最後回傳：
       - hard_users 清單
       - E_add_promote：實際加的邊 tensor
       - E_remove_suppress：實際減掉的邊 tensor
       - target_train_new：加減完之後的新 target_train_edge_index
    """

    def __init__(self, top_ratio, log_dir="logs/hard_user"):
        """
        建構子

        參數：
            top_ratio: float
                從 GroupB 中要挑出多少比例的使用者當 Hard Users。
                - 例如 top_ratio=0.10 表示挑 GroupB 中距離最大的前 10%。
                - 注意：這裡仍然是「排序 + 取前 K」，但沒有任何隨機成分。

            log_dir: str
                用來存 log 與 .npy 檔的資料夾路徑。
        """
        self.top_ratio = top_ratio
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    # ----------------------------------------------------
    # 1. 根據冷門商品切出 GroupA / GroupB
    # ----------------------------------------------------
    @staticmethod
    def _split_users_by_target_item(target_train_local, cold_item_local, num_users):
        """
        根據「指定的冷門 target item (local id)」把 user 分成兩群：

            - GroupA：有買過 cold_item_local 的 user
            - GroupB：沒買過 cold_item_local 的 user

        這裡的 target_train_local 是「local item id」版：
            - user id：0 ~ num_users-1
            - item id：0 ~ num_target_items-1

        參數：
            target_train_local: torch.LongTensor, shape=[2, E]
                target domain 的 train 邊（item 已轉成本地編號）
            cold_item_local: int
                冷門商品在 target domain 的 local id
            num_users: int
                使用者數量（假設 user id 範圍為 [0, num_users-1]）

        回傳：
            groupA: list[int]
                有買過冷門商品的 user id
            groupB: list[int]
                沒買過冷門商品的 user id
        """
        # 1. 找出所有邊中 item == cold_item_local 的位置
        mask = (target_train_local[1] == cold_item_local)

        # 2. 取出對應的 user，並 unique，得到有買冷門商品的 user 集合
        ua = target_train_local[0][mask].unique()
        groupA = set(ua.tolist())

        # 3. 所有 user id = {0, 1, ..., num_users-1}
        all_users = set(range(num_users))

        # 4. GroupB = 全體 user - groupA
        groupB = list(all_users - groupA)

        return list(groupA), groupB

    # ----------------------------------------------------
    # 2. 從 GroupB 中挑 Hard Users
    # ----------------------------------------------------
    @staticmethod
    def _pick_hard_users(user_emb_target, groupA, groupB, top_ratio):
        """
        從 groupB 中選出「Hard Users」：即對 groupA 使用者最不相似的那群人。

        直覺解釋：
            - GroupA = 已經有買冷門商品的 user
            - 我們想找的 Hard User = 那些離 GroupA「最遠」的人
            - 這些人如果被推去買冷門商品，算是比較「困難」的對象

        方法：
            1. 從 user_emb_target 中抓出 groupA、groupB 的 embedding，做 L2 normalize
            2. 計算 sim = uB @ uA^T（cos similarity）
            3. 對每個 groupB user 取 max_sim（對所有 groupA 的最大相似度）
            4. 定義 dist = 1 - max_sim，dist 越大表示越不相似
            5. 按 dist 由大到小排序，取前 top_ratio 比例當 Hard Users

        參數：
            user_emb_target: torch.FloatTensor, shape=[num_users, dim]
                target domain 的 user embedding（例如 SGL 訓練結果）
            groupA: list[int]
                有買冷門商品的 user id 清單
            groupB: list[int]
                沒買冷門商品的 user id 清單
            top_ratio: float
                從 groupB 中取多少比例當 Hard Users

        回傳：
            hard_users: list[int]
                被選為 Hard User 的 user id 列表
        """
        # 任一群為空 → 無法計算距離，直接回傳空
        if len(groupA) == 0 or len(groupB) == 0:
            return []

        # 轉成 tensor，方便到 embedding 做 index
        A = torch.tensor(groupA, device=user_emb_target.device)
        B = torch.tensor(groupB, device=user_emb_target.device)

        # 抓出對應的 embedding 並做 L2 normalize
        uA = F.normalize(user_emb_target[A], dim=-1)  # shape=[|A|, dim]
        uB = F.normalize(user_emb_target[B], dim=-1)  # shape=[|B|, dim]

        # sim[b, a] = uB[b] · uA[a]，cos similarity
        sim = torch.matmul(uB, uA.t())  # shape=[|B|, |A|]

        # 每個 groupB user 對 groupA user 的最大相似度
        max_sim, _ = sim.max(dim=1)     # shape=[|B|]

        # 距離定義為 1 - 最大相似度
        dist = 1 - max_sim

        # 根據 top_ratio 決定要挑多少人當 Hard Users：
        k = math.floor(len(groupB) * top_ratio)

        # 🔒 確保至少取 1 人（除非 groupB 為 0）
        if k <= 0:
            k = 1

        # 防止 k > groupB 人數
        k = min(k, len(groupB))

        # torch.topk 取出距離最大的前 k 個 index
        top_idx = torch.topk(dist, k=k, largest=True).indices

        # 把這些 index 對應回原本的 user id（注意 B 是 groupB 的 user id）
        return [int(B[i]) for i in top_idx]

    # ----------------------------------------------------
    # 3. 找出 target domain 的熱門商品
    # ----------------------------------------------------
    @staticmethod
    def _get_popular_items(target_train, num_users, num_source_items, popular_top_k):
        """
        根據 target_train_edge_index 中的出現頻率，選出 target domain 的熱門商品。

        注意 global id 編號規則（常見設定）：
            - user id           : [0, num_users-1]
            - source item id    : [num_users, num_users+num_source_items-1]
            - target item id    : [num_users+num_source_items, ... ]

        我們只想挑出「target item 的熱門商品」，因此：
            - 只保留 global_item_id >= num_users + num_source_items

        步驟：
            1. 對 target_train[1] 的 item id 做 unique + 計數
            2. 依照出現次數由大到小排序
            3. 過濾掉非 target item
            4. 取前 popular_top_k 個

        參數：
            target_train: torch.LongTensor, shape=[2, E]
                target domain 的 train 邊（global id）
            num_users: int
                user 數量
            num_source_items: int
                source domain item 數量
            popular_top_k: int
                要挑出幾個最熱門的 target item

        回傳：
            popular_items: list[int]
                最熱門的 target item（global id），最多 popular_top_k 個
        """
        # 抓出所有 item id（global）
        item_ids = target_train[1]

        # unique_items: 所有出現過的 item
        # counts: 各 item 的出現次數
        uniq_items, counts = item_ids.unique(return_counts=True)

        # 依照 counts 由大到小排序
        order = torch.argsort(counts, descending=True)
        sorted_items = uniq_items[order].tolist()

        # target item 的 global 編號下界
        target_min = num_users + num_source_items

        # 只保留 target item，並取前 popular_top_k 個
        popular_items = [i for i in sorted_items if i >= target_min][:popular_top_k]
        return popular_items

    # ----------------------------------------------------
    # 4. 主流程：執行 Hard User 加邊 + 減邊
    # ----------------------------------------------------
    def run(
        self,
        split_result,
        user_emb_target,
        num_users,
        num_source_items,
        num_target_items,
        cold_item_id,      # 冷門商品的 global id（在 target domain）
        popular_top_k,     # 要挑出幾個 popular items 當「抑制池」
    ):
        """
        主函式：執行 Hard User 的「加邊 + 減邊」策略。

        設計重點：
            - Hard Users 一旦被選中 → 一律加 promoted 冷門商品邊
            - Hard Users × popular_items 的既有邊 → 一律刪掉
            - 完全沒有隨機比例、沒有 randomness，結果 deterministic

        參數：
            split_result: dict
                典型內容：
                {
                    "source_train_edge_index": Tensor([2, E_s]),
                    "target_train_edge_index": Tensor([2, E_t]),
                    "target_valid_edge_index": ...,
                    "target_test_edge_index":  ...
                }
                此函式只會修改 "target_train_edge_index"。

            user_emb_target: torch.FloatTensor, shape=[num_users, dim]
                target domain 的 user embedding（例如模型輸出）

            num_users: int
            num_source_items: int
            num_target_items: int
                用來判斷 id 範圍與 local/global 轉換

            cold_item_id: int
                target domain 冷門商品的 global id

            popular_top_k: int
                作為 popular item pool 的大小。
                - 例如 popular_top_k=100 表示選出最熱門的 100 個 target item，作為「要被抑制的商品池」。

        回傳：
            result: dict
                {
                    "hard_users": list[int],
                    "E_add_promote": Tensor([2, #added]),
                    "E_remove_suppress": Tensor([2, #removed]),
                    "target_train_new": Tensor([2, E_new]),
                }
        """
        logging.info("🔥 [HardUser-Clean] 執行 Hard User 加邊 + 減邊（全 Hard User 參與）")

        # 取出 target domain 的 train 邊（global id）
        target_train_edge_index = split_result["target_train_edge_index"].clone()

        # ------------------------------------------------
        # 4-1. 將冷門商品 global id 轉成本地 local id
        # ------------------------------------------------
        cold_item_global = cold_item_id
        # local = global - (num_users + num_source_items)
        cold_item_local = cold_item_global - (num_users + num_source_items)

        # 防呆檢查：冷門商品 local id 必須落在 [0, num_target_items-1]
        assert 0 <= cold_item_local < num_target_items, \
            f"cold_item_local={cold_item_local} 超出 [0, {num_target_items-1}] 範圍，請檢查 cold_item_id / num_users / num_source_items / num_target_items"

        # 產出 local 版的 target_train_edge_index：
        #   user: 保持不動（0~num_users-1）
        #   item: 減掉 offset 變成 0~num_target_items-1
        target_train_local = target_train_edge_index.clone()
        target_train_local[1] -= (num_users + num_source_items)

        # ------------------------------------------------
        # 4-2. 根據冷門商品切 GroupA / GroupB
        # ------------------------------------------------
        groupA, groupB = self._split_users_by_target_item(
            target_train_local,
            cold_item_local,
            num_users
        )
        logging.info(f"[HardUser] GroupA={len(groupA)} (有買冷門), GroupB={len(groupB)} (沒買冷門)")
        # ⭐⭐⭐ 新增：印出所有 GroupA user ⭐⭐⭐
        print("\n[HardUser] === GroupA (有買冷門商品的 users) ===")
        for u in sorted(groupA):
            print(f"  user {u}")
        print(f"[HardUser] GroupA user list printed ({len(groupA)} users)\n")
        
        print("\n=== DEBUG: Who actually bought the cold item? ===")
        cold_item_global = cold_item_id
        count = 0
        for u, i in split_result["target_train_edge_index"].t().tolist():
            if i == cold_item_global:
                print(f"user {u} bought cold_item {cold_item_global}")
                count += 1

        print(f"Total = {count} users")

        # ------------------------------------------------
        # 4-3. 從 GroupB 中選 Hard Users
        # ------------------------------------------------
        hard_users = self._pick_hard_users(
            user_emb_target,
            groupA,
            groupB,
            self.top_ratio
        )
        logging.info(f"[HardUser] Hard Users 數量={len(hard_users)} (top_ratio={self.top_ratio})")

        # 沒有 Hard User → 不做任何修改，直接回傳原圖
        if len(hard_users) == 0:
            logging.warning("⚠ [HardUser] 無 Hard Users，直接回傳原始 target_train_edge_index")
            return {
                "hard_users": [],
                "E_add_promote": torch.empty((2, 0), dtype=torch.long),
                "E_remove_suppress": torch.empty((2, 0), dtype=torch.long),
                "target_train_new": target_train_edge_index
            }

        # ------------------------------------------------
        # 4-4. 對所有 Hard Users 加「冷門商品」邊 (promote)
        # ------------------------------------------------
        promote_edges = [(u, cold_item_global) for u in hard_users]
        promote_edges = torch.tensor(promote_edges, dtype=torch.long).t()

        logging.info(f"[HardUser] 加 promoted item 的邊數：{promote_edges.size(1)}")
        print("\n[HardUser] === 加邊（promote cold item） ===")
        print(f"加邊總數：{promote_edges.size(1)}")
        for u, i in promote_edges.t().tolist():
            print(f"  + user {u} -> item {i}")

        # ------------------------------------------------
        # 4-5. Popular item pool（target domain 熱門商品）
        # ------------------------------------------------
        popular_items = self._get_popular_items(
            target_train_edge_index,
            num_users,
            num_source_items,
            popular_top_k
        )

        print("\n==================== Popular Item 統計 ====================")
        print(f"Top-{popular_top_k} popular items（global id）:")
        print(popular_items)

        # 統計所有 user 與 Hard User 的購買次數
        all_items = target_train_edge_index[1].tolist()
        all_users = target_train_edge_index[0].tolist()

        popular_stats = {i: {"all_user": 0, "hard_user": 0} for i in popular_items}
        hard_user_set = set(hard_users)

        for u, i in zip(all_users, all_items):
            if i in popular_stats:
                popular_stats[i]["all_user"] += 1
                if u in hard_user_set:
                    popular_stats[i]["hard_user"] += 1

        # ======== 加入累積欄位版本 ========
        cumulative_all = 0
        cumulative_hard = 0

        print("\n📊 Popular item 出現統計（含累積）：")
        print("(Item, 全體 user 次數, Hard Users 次數, 全體累積, Hard累積)")

        for item in popular_items:
            st = popular_stats[item]

            cumulative_all += st["all_user"]
            cumulative_hard += st["hard_user"]

            print(
                f"Item {item}: "
                f"all_user={st['all_user']}, "
                f"hard_user={st['hard_user']}, "
                f"cumulative_all={cumulative_all}, "
                f"cumulative_hard={cumulative_hard}"
            )


        # ------------------------------------------------
        # 4-6. 尋找 Hard User × popular items 的 existing edges → 全部刪掉
        # ------------------------------------------------
        exist_set = _tensor2set(target_train_edge_index)

        remove_edges = []
        for u in hard_users:
            for i in popular_items:
                if (u, i) in exist_set:
                    remove_edges.append((u, i))

        if len(remove_edges):
            remove_edges = torch.tensor(remove_edges, dtype=torch.long).t()
        else:
            remove_edges = torch.empty((2, 0), dtype=torch.long)

        logging.info(f"[HardUser] 減邊數量：{remove_edges.size(1)}")
        print("\n[HardUser] === 減邊（suppress popular item） ===")
        print(f"減邊總數：{remove_edges.size(1)}")
        for u, i in remove_edges.t().tolist():
            print(f"  - user {u} -> item {i}")

        # ------------------------------------------------
        # 4-7. 套用（先減邊，再加邊）
        # ------------------------------------------------
        new_edge = target_train_edge_index
        if remove_edges.numel():
            new_edge = _apply_remove(new_edge, remove_edges.t().tolist())
        if promote_edges.numel():
            new_edge = _apply_add(new_edge, promote_edges.t().tolist())

        logging.info(
            f"[HardUser] target_train_edge_index: 原本 {target_train_edge_index.size(1)} 條 → "
            f"現在 {new_edge.size(1)} 條"
        )

        # ------------------------------------------------
        # 4-8. 存 .npy 檔方便 debug / 分析
        # ------------------------------------------------
        np.save(os.path.join(self.log_dir, "E_add_promote.npy"), promote_edges.cpu().numpy())
        np.save(os.path.join(self.log_dir, "E_remove_suppress.npy"), remove_edges.cpu().numpy())
        np.save(os.path.join(self.log_dir, "target_train_new.npy"), new_edge.cpu().numpy())

        # ------------------------------------------------
        # 4-9. 統一回傳結果
        # ------------------------------------------------
        return {
            "hard_users": hard_users,
            "E_add_promote": promote_edges,
            "E_remove_suppress": remove_edges,
            "target_train_new": new_edge,
        }
