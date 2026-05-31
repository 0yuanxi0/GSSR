import torch
import numpy as np
import math


class SemanticAnalyzer:
    """语义塔贡献分析: α vs 用户稀疏度, 高α/低α 分组对比"""

    def __init__(self, data, dataset_name):
        self.train_gt = data['train_gt']
        self.test_gt = data['test_gt']
        self.n_users = data['n_users']
        self.n_items = data['n_items']
        self.device = data['device']
        self.dataset_name = dataset_name.lower()

        self.user_degrees = np.zeros(self.n_users + 1)
        for u_str, items in self.train_gt.items():
            u_id = int(u_str)
            if u_id <= self.n_users:
                self.user_degrees[u_id] = len(items)

    def run_eval(self, model, k_list=[5, 10, 20, 50], batch_size=1024, verbose=True):
        """按 α 分组评测: 语义塔贡献越高的用户, 推荐性能是否越好"""
        model.eval()
        max_k = max(k_list)

        with torch.no_grad():
            out = model.forward()
            all_users = out['final_user']
            all_items = out['final_item']
            gate_user = out.get('gate_user', None)

        test_users = [int(u) for u in self.test_gt.keys() if int(u) <= self.n_users]

        # 获取每用户的 α, 分三组
        if gate_user is not None:
            gate_np = gate_user.detach().cpu().numpy()
        else:
            gate_np = np.zeros(self.n_users + 1)

        high_a, mid_a, low_a = [], [], []
        for u in test_users:
            a = gate_np[u]
            if a > 0.1:
                high_a.append(u)
            elif a > 0.01:
                mid_a.append(u)
            else:
                low_a.append(u)

        # 按训练购买次数绝对区间分组
        degrees = [self.user_degrees[u] for u in test_users]
        sparse_ranges = [
            ('Ultra-Sparse [1,2]',  lambda d: 1 <= d <= 2),
            ('Sparse [3,7]',        lambda d: 3 <= d <= 7),
            ('Moderate [8,15]',     lambda d: 8 <= d <= 15),
            ('Active [16,30]',      lambda d: 16 <= d <= 30),
            ('Super-Active [31+]',  lambda d: d >= 31),
        ]
        sparse_users = {}
        for gname, gfunc in sparse_ranges:
            sparse_users[gname] = [u for u, d in zip(test_users, degrees) if gfunc(d)]

        groups = {
            'High α (>0.10)': high_a,
            'Mid α (0.01~0.10)': mid_a,
            'Low α (<0.01)': low_a,
        }

        alpha_results = {}
        for gname, ulist in groups.items():
            if not ulist: continue
            m = self._compute_metrics(model, all_users, all_items, ulist, k_list, max_k, batch_size)
            alpha_results[gname] = {'n_users': len(ulist), **m}
            # α 统计
            alphas = [gate_np[u] for u in ulist]
            alpha_results[gname]['α_mean'] = float(np.mean(alphas))
            alpha_results[gname]['α_median'] = float(np.median(alphas))

        sparsity_results = {}
        for gname, ulist in sparse_users.items():
            if not ulist: continue
            m = self._compute_metrics(model, all_users, all_items, ulist, k_list, max_k, batch_size)
            sparsity_results[gname] = {'n_users': len(ulist), **m}
            alphas = [gate_np[u] for u in ulist]
            sparsity_results[gname]['alpha_mean'] = float(np.mean(alphas))

        # 全部用户也加入
        m_all = self._compute_metrics(model, all_users, all_items, test_users, k_list, max_k, batch_size)
        sparsity_results['All Users'] = {'n_users': len(test_users), **m_all}

        if verbose:
            self._print_alpha_table(alpha_results, k_list)
            self._print_sparsity_table(sparsity_results, k_list)

        return {'alpha_groups': alpha_results, 'sparsity_groups': sparsity_results}

    def _compute_metrics(self, model, all_users, all_items, user_list, k_list, max_k, batch_size):
        metrics_sum = {k: {'Recall': 0.0, 'NDCG': 0.0} for k in k_list}
        n = len(user_list)
        for i in range(0, n, batch_size):
            batch = user_list[i:i+batch_size]
            u_emb = all_users[batch]
            scores = torch.matmul(u_emb, all_items.t())
            for idx, u_id in enumerate(batch):
                train_items = self.train_gt.get(str(u_id), [])
                if train_items:
                    scores[idx, train_items] = -float("inf")
            _, topk_idx = torch.topk(scores, max_k, dim=1)
            topk_idx = topk_idx.cpu().numpy()
            for idx, u_id in enumerate(batch):
                gt = set(self.test_gt.get(str(u_id), []))
                if not gt: continue
                pred_list = list(topk_idx[idx])
                for k in k_list:
                    hits = len(set(pred_list[:k]) & gt)
                    metrics_sum[k]['Recall'] += hits / len(gt)
                    dcg = sum(1.0/math.log2(j+2) for j, it in enumerate(pred_list[:k]) if it in gt)
                    idcg = sum(1.0/math.log2(j+2) for j in range(min(len(gt), k)))
                    metrics_sum[k]['NDCG'] += dcg/idcg if idcg > 0 else 0

        result = {}
        for k in k_list:
            result[f'R@{k}'] = metrics_sum[k]['Recall'] / n
            result[f'N@{k}'] = metrics_sum[k]['NDCG'] / n
        return result

    def _print_alpha_table(self, results, k_list):
        print(f"\n{'='*40} α-Group Analysis {'='*40}")
        print(f"Dataset: {self.dataset_name.upper()}")
        header = f"{'Group':<22} | {'Users':<6} | {'α_mean':>6}"
        for k in k_list:
            header += f" | R@{k:<2}   | N@{k:<2}  "
        print(header)
        print("-" * len(header))
        for gname in results:
            r = results[gname]
            line = f"{gname:<22} | {r['n_users']:<6} | {r['α_mean']:6.3f}"
            for k in k_list:
                line += f" | {r.get(f'R@{k}', 0):.4f} | {r.get(f'N@{k}', 0):.4f}"
            print(line)

    def _print_sparsity_table(self, results, k_list):
        print(f"\n{'='*70} Sparsity Analysis (by Purchase Count) {'='*70}")
        print(f"Dataset: {self.dataset_name.upper()}")
        header = f"{'Group':<28} | {'Users':>6} | {'α_mean':>6}"
        for k in k_list:
            header += f" | R@{k:<2}   | N@{k:<2}  "
        print(header)
        print("-" * len(header))

        order = ['Ultra-Sparse [1,2]', 'Sparse [3,7]', 'Moderate [8,15]',
                 'Active [16,30]', 'Super-Active [31+]', 'All Users']
        for gname in order:
            if gname not in results: continue
            r = results[gname]
            line = f"{gname:<28} | {r['n_users']:>6} | {r['alpha_mean']:6.4f}"
            for k in k_list:
                line += f" | {r.get(f'R@{k}', 0):.4f} | {r.get(f'N@{k}', 0):.4f}"
            print(line)
