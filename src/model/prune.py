# src/model/prune.py
# Structure-Evolving Memory Pruner (DeMBR-style, simplified)

import torch
import torch.nn.functional as F


@torch.no_grad()
def _split_user_item(behavior_emb, n_users, n_items):
    user_emb = behavior_emb[:n_users + 1]
    item_emb = behavior_emb[n_users + 1:n_users + 1 + n_items + 1]
    return user_emb, item_emb


def _edge_user_item(edge_index, n_users):
    src, dst = edge_index
    mask = (src <= n_users) & (dst > n_users)
    u = src[mask]
    i0 = (dst[mask] - (n_users + 1)) - 1
    return u.long().view(-1), i0.long().view(-1), mask


class LightweightPruner:
    def __init__(self, model, data,
                 thresh: float = 0.10,
                 use_quantile: float = None,
                 q_schedule=None,
                 temp: float = 0.0,
                 warmup_epochs: int = 8,
                 prune_every: int = 2,
                 total_epochs: int = None,
                 tail_freeze: int = 8,
                 prune_buy: bool = False,
                 buy_quantile: float = None,
                 buy_thresh: float = None,
                 buy_drop_ratio: float = 0.01,
                 min_deg_item: int = 3,
                 max_drop_ratio: float = 0.05,
                 m_protect: int = 3,
                 mm_chunk: int = 8192,
                 device=None,
                 **kwargs):

        self.model = model
        self.data = data
        self.device = device

        self.n_users = data['n_users']
        self.n_items = data['n_items']
        self.bsg_types = data['bsg_types']
        self.prune_types = self.bsg_types

        self.use_quantile = float(use_quantile) if use_quantile is not None else None
        self.thresh = None if self.use_quantile is not None else float(thresh)
        self.q_schedule = list(q_schedule) if q_schedule else None
        self.round_idx = 0

        self.temp = float(temp)

        self.warmup_epochs = int(warmup_epochs)
        self.prune_every = int(prune_every)
        self.total_epochs = int(total_epochs) if total_epochs is not None else None
        self.tail_freeze = int(tail_freeze)

        self.prune_buy = bool(prune_buy)
        self.buy_quantile = float(buy_quantile) if buy_quantile is not None else None
        self.buy_thresh = float(buy_thresh) if buy_thresh is not None else None
        self.buy_drop_ratio = float(buy_drop_ratio)

        self.min_deg_item = int(min_deg_item)
        self.max_drop_ratio = float(max_drop_ratio)
        self.m_protect = int(m_protect)
        self.mm_chunk = int(mm_chunk)

        U, I = self.n_users, self.n_items
        self.memory = {
            b: torch.zeros((U, I), dtype=torch.float32)
            for b in self.prune_types
        }

    @torch.no_grad()
    def maybe_prune(self, epoch: int):
        if self.total_epochs is not None and epoch >= self.total_epochs - self.tail_freeze:
            if epoch == self.total_epochs - self.tail_freeze:
                print(f"[PRUNE] freeze graph from epoch {epoch}")
            return

        if epoch < self.warmup_epochs:
            return
        if (epoch - self.warmup_epochs) % self.prune_every != 0:
            return

        print(f"[PRUNE] epoch={epoch}")
        self.model.eval()
        emb_dict = self.model.forward()

        for b in self.prune_types:
            if b.lower() == "buy" and not self.prune_buy:
                continue
            self._update_memory_and_prune_one(b, emb_dict[b])
        self.round_idx += 1

    @torch.no_grad()
    def _pick_threshold(self, M, behavior_type):
        b = behavior_type.lower()
        if b == "buy" and self.prune_buy:
            if self.buy_quantile is not None:
                flat = M.view(-1)
                k = max(1, int(flat.numel() * self.buy_quantile))
                return float(torch.kthvalue(flat, k).values.item())
            if self.buy_thresh is not None:
                return float(self.buy_thresh)

        if self.use_quantile is not None:
            q_now = self.use_quantile
            if self.q_schedule:
                q_now = self.q_schedule[min(self.round_idx, len(self.q_schedule) - 1)]
            flat = M.view(-1)
            k = max(1, int(flat.numel() * q_now))
            return float(torch.kthvalue(flat, k).values.item())

        return float(self.thresh)

    @torch.no_grad()
    def _update_memory_and_prune_one(self, behavior_type, behavior_emb):
        dev = self.device or behavior_emb.device
        U, I = self.n_users, self.n_items

        u_all, i_all = _split_user_item(behavior_emb, U, I)
        u_all = F.normalize(u_all, dim=1)
        i_all = F.normalize(i_all, dim=1)
        Ue = u_all[1:]
        Ie = i_all[1:]

        S_cpu = torch.empty((U, I), dtype=torch.float32)
        bs = self.mm_chunk
        Ue_dev = Ue.to(dev, non_blocking=True)

        for j0 in range(0, I, bs):
            j1 = min(j0 + bs, I)
            Ie_blk = Ie[j0:j1].to(dev, non_blocking=True)
            S_blk = (Ue_dev @ Ie_blk.t()).to('cpu', dtype=torch.float32)
            if self.temp and self.temp > 0:
                S_blk = torch.tanh(S_blk / self.temp)
            S_cpu[:, j0:j1] = S_blk

        M = self.memory[behavior_type]
        beta = 0.7 if self.round_idx < 2 else 0.5
        M.mul_(1.0 - beta).add_(beta * S_cpu)

        tau = self._pick_threshold(M, behavior_type)

        edge_index = self.model.edge_dict[behavior_type]
        u_idx, i_idx0, _ = _edge_user_item(edge_index, U)
        if u_idx.numel() == 0:
            return

        u0 = (u_idx - 1).cpu()
        i0 = i_idx0.cpu()

        keep_dir_cpu = (M[u0, i0] >= tau)
        keep_dir_dev = keep_dir_cpu.to(u_idx.device)

        # top-m protection: 向量化, 每item保留分数最高的m条边
        if self.m_protect > 0:
            # 用 topk 一次性计算每 item 的 top-m 候选
            i0_dev = i_idx0.to(u_idx.device)
            vals_dev = M[u0, i0].to(u_idx.device)  # 从CPU内存直接读
            # 对每个item, 找到其所有边的 score, 保留 top-m
            protect = torch.zeros_like(keep_dir_dev)
            for it in (i0_dev.unique()):  # 只迭代有边的item, 而非全I
                pos = (i0_dev == it).nonzero(as_tuple=False).view(-1)
                if pos.numel() <= self.m_protect:
                    protect[pos] = True
                else:
                    pos_vals = vals_dev[pos]
                    _, topk_pos = torch.topk(pos_vals, k=self.m_protect, largest=True)
                    protect[pos[topk_pos]] = True
            keep_dir_dev |= protect

        # safety guards
        i_counts = torch.bincount(i_idx0, minlength=I)

        if behavior_type.lower() == "buy" and self.prune_buy:
            drop_cap = self.buy_drop_ratio
        else:
            growth = 0.5
            drop_cap = self.max_drop_ratio * (1.0 + growth * self.round_idx)
            drop_cap = min(drop_cap, 0.30)

        allow = torch.minimum(
            (drop_cap * i_counts.float()).floor().long(),
            (i_counts - self.min_deg_item).clamp_min(0)
        )

        cand = (~keep_dir_dev).nonzero(as_tuple=False).view(-1)
        if cand.numel() > 0:
            it = i_idx0[cand]
            order = torch.argsort(it)
            cand = cand[order]
            it = it[order]

            dcnt = torch.bincount(it, minlength=I)
            dstart = torch.cumsum(
                torch.cat([dcnt.new_zeros(1), dcnt[:-1]]), 0
            )

            take = torch.zeros(cand.numel(), dtype=torch.bool, device=u_idx.device)
            for i in (dcnt > 0).nonzero(as_tuple=False).view(-1).tolist():
                a = int(allow[i].item())
                if a <= 0:
                    continue
                s = int(dstart[i].item())
                e = s + int(dcnt[i].item())
                take[s:s + min(a, e - s)] = True

            drop_final = cand[take]
            keep_mask = torch.ones(u_idx.size(0), dtype=torch.bool, device=u_idx.device)
            keep_mask[drop_final] = False
            kept_pos = keep_mask.nonzero(as_tuple=False).view(-1)
        else:
            kept_pos = torch.arange(u_idx.size(0), device=u_idx.device)

        kept_u = u_idx[kept_pos]
        kept_i = i_idx0[kept_pos] + 1
        src_u = kept_u
        dst_i = kept_i + (U + 1)
        new_edge = torch.stack([src_u, dst_i], dim=0)
        new_edge = torch.cat([new_edge, new_edge.flip(0)], dim=1)

        target_dev = self.model.user_embedding.weight.device
        self.model.edge_dict[behavior_type] = new_edge.to(target_dev, non_blocking=True)

        if 'edge_dict' in self.data:
            self.data['edge_dict'][behavior_type] = self.model.edge_dict[behavior_type]

        print(f"[PRUNE] [{behavior_type}] new edges (undirected) = {new_edge.size(1)}")
