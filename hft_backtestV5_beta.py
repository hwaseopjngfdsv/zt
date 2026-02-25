# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 17:48:21 2025

@author: fyx90

tradesV6 , 无撤单直接成交，都是炸板，202403 亏20%
tradesV7  不包括一字板
tradesV8 , 叠加新股池 + 计算整体队列是否会被炸板
tradesV9, 叠加新股池,分为强势股和非强势股，下单延迟为10ms，强势股安全期1分钟，立刻打板，非强势股安全期5分钟，按正常的参数下单
"""


import os 
os.chdir(fr'F:\tools')
from hft_backtest_utils import *

from collections import OrderedDict
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Set, Union

from collections import defaultdict
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any


# ===== 时间转换（整数 HHMMSSmmm <-> 毫秒，自午夜起）=====
def _t_to_ms(t: int) -> int:
    t = int(t)
    h  = t // 10000000
    r1 = t %  10000000
    m  = r1 // 100000
    r2 = r1 %  100000
    s  = r2 // 1000
    ms = r2 %  1000
    return ((h*3600 + m*60 + s) * 1000) + ms

def _ms_to_t(ms: np.ndarray) -> np.ndarray:
    ms = ms % (24*3600*1000)
    h  = ms // (3600*1000)
    r1 = ms %  (3600*1000)
    m  = r1 // (60*1000)
    r2 = r1 %  (60*1000)
    s  = r2 // 1000
    mm = r2 %  1000
    return (h*10000000 + m*100000 + s*1000 + mm).astype(np.int64)

# ===== 10ms 补桶：懒初始化 + 增量更新 =====
def pad_10ms_lazy(
    state_pad: Optional[Dict[str, Any]],
    inc_df: Optional[pd.DataFrame],
    *,
    interval_ms: int = 10,
    base_start_time: Optional[int] = None,  # 建议传 first_over_60，作为填桶起点
    ticker_col: str = "Ticker",
    cols_cancel: str = "成交总量_撤单",
    cols_trade: str = "成交总量",
    col_total: str  = "总量",
    col_total_adj: str = "总量调整",
) -> (Dict[str, Any], pd.DataFrame):
    """
    增量补齐 10ms 桶。
    - state_pad: 第一次可为 None（懒初始化）
    - inc_df: 本批新增的“有量桶”，至少包含 start_time（HHMMSSmmm）。体量列缺失会按 0 处理。
    返回: (state_pad, inc_padded)
      - state_pad['df']: 全量“已补齐”的结果（共享持久）
      - inc_padded: 本批“新增区间内”的完整 10ms 桶（含补零后的行）
    """
    # —— 1) 标准化增量输入 —— #
    if inc_df is None or inc_df.empty:
        # 没有新桶：若已初始化则返回一个空 inc_padded
        if state_pad is None:
            if base_start_time is None:
                raise ValueError("首次调用需要 inc_df 非空，或提供 base_start_time 用于初始化。")
            # 懒初始化成空状态（仅确定起点）
            start_ms0 = (_t_to_ms(base_start_time) // interval_ms) * interval_ms
            df0 = pd.DataFrame(columns=["start_ms","start_time","end_time",
                                        cols_cancel, cols_trade, col_total, ticker_col, col_total_adj])
            df0 = df0.astype({
                "start_ms":"int64","start_time":"int64","end_time":"int64",
                cols_cancel:"int64", cols_trade:"int64", col_total:"int64",
                col_total_adj:"int64"
            }, errors="ignore")
            state_pad = {"df": df0.set_index("start_ms"), "last_ms": start_ms0 - interval_ms,
                         "interval_ms": int(interval_ms), "ticker": None}
            return state_pad, df0.iloc[0:0].copy()
        else:
            # 已有状态但无增量
            return state_pad, state_pad["df"].loc[[]].reset_index()[["start_time","end_time",cols_cancel,cols_trade,col_total,ticker_col,col_total_adj]]

    x = inc_df.copy()
    if "start_time" not in x.columns:
        raise KeyError("inc_df 必须包含列 'start_time'（HHMMSSmmm）")
    # 体量列缺失按 0
    for c in [cols_cancel, cols_trade, col_total, col_total_adj, ticker_col]:
        if c not in x.columns:
            if c == ticker_col:
                x[c] = None
            else:
                x[c] = 0

    # 计算本批的 start_ms（桶起点，毫秒）
    inc_ms = x["start_time"].astype("int64").map(_t_to_ms)
    inc_ms = (inc_ms // interval_ms) * interval_ms
    x = x.copy()
    x["start_ms"] = inc_ms

    # —— 2) 懒初始化状态 —— #
    if state_pad is None:
        if base_start_time is not None:
            start_ms0 = (_t_to_ms(base_start_time) // interval_ms) * interval_ms
        else:
            start_ms0 = int(inc_ms.min())
        df0 = pd.DataFrame(columns=["start_ms","start_time","end_time",
                                    cols_cancel, cols_trade, col_total, ticker_col, col_total_adj])
        state_pad = {
            "df": df0.set_index("start_ms"),
            "last_ms": start_ms0 - interval_ms,     # 表示还没补到 start_ms0
            "interval_ms": int(interval_ms),
            "ticker": x[ticker_col].dropna().iloc[0] if x[ticker_col].notna().any() else None
        }

    df = state_pad["df"]
    last_ms = int(state_pad["last_ms"])
    tick = state_pad["ticker"]
    inter = int(state_pad["interval_ms"])

    # —— 3) 计算需要补充的区间（只往后补） —— #
    new_max_ms = int(inc_ms.max())
    if new_max_ms > last_ms:
        pads = np.arange(last_ms + inter, new_max_ms + 1, inter, dtype=np.int64)
        if pads.size:
            pads_df = pd.DataFrame({
                "start_ms": pads,
                "start_time": _ms_to_t(pads),
                "end_time":   _ms_to_t(pads + inter),
                cols_cancel:  0,
                cols_trade:   0,
                col_total:    0,
                ticker_col:   tick
            })
            pads_df[col_total_adj] = pads_df[col_total]  # 没有平滑则等于总量
            pads_df = pads_df.set_index("start_ms")
            # 追加补零行
            df = pd.concat([df, pads_df], axis=0)
            # 排序去重（理论上 pads 不会与已有重叠，这里稳妥起见）
            df = df[~df.index.duplicated(keep="last")].sort_index()

    # —— 4) 用本批 inc 覆盖对应桶（把“非零量”写进来） —— #
    upd = x[[ "start_ms","start_time",ticker_col, cols_cancel, cols_trade]]
    upd = upd.set_index("start_ms")
    # 计算总量 & 总量调整（如果 inc 没提供）
    upd[col_total] = pd.to_numeric(upd[cols_cancel], errors="coerce").fillna(0).astype("int64") + \
                     pd.to_numeric(upd[cols_trade],  errors="coerce").fillna(0).astype("int64")
    if col_total_adj in x.columns and x[col_total_adj].notna().any():
        # 用户传了总量调整，就用之（没有的行我们后面再补）
        adj = x.set_index("start_ms")[col_total_adj]
        upd[col_total_adj] = pd.to_numeric(adj, errors="coerce")
    else:
        upd[col_total_adj] = upd[col_total]

    # 覆盖写入（upsert）
    # 先确保所有待更新的 start_ms 行存在（补零时已包含大多数；若 inc 出现“过去”的 ms，这里也能补上）
    missing_ms = upd.index.difference(df.index)
    if len(missing_ms):
        extra = pd.DataFrame(index=missing_ms)
        extra["start_time"] = _ms_to_t(missing_ms.values)
        extra["end_time"]   = _ms_to_t(missing_ms.values + inter)
        for c in [cols_cancel, cols_trade, col_total, col_total_adj]:
            extra[c] = 0
        extra[ticker_col] = tick
        df = pd.concat([df, extra], axis=0).sort_index()

    df.update(upd)  # 量和 start_time/ticker 以 inc 为准

    # 总量调整缺失的行补为总量
    df[col_total_adj] = pd.to_numeric(df[col_total_adj], errors="coerce")
    df[col_total_adj] = df[col_total_adj].where(~df[col_total_adj].isna(), df[col_total])

    # —— 5) 写回状态，返回“本批新增区间”的完整行 —— #
    state_pad["df"] = df
    state_pad["last_ms"] = max(last_ms, new_max_ms)
    # 若本批带有 Ticker，更新默认 ticker
    if x[ticker_col].notna().any():
        state_pad["ticker"] = x[ticker_col].dropna().iloc[0]

    # 本批“新增区间”的所有桶（pads ∪ inc_ms）
    updated_ms = np.unique(np.concatenate([
        np.arange(last_ms + inter, new_max_ms + 1, inter, dtype=np.int64) if new_max_ms > last_ms else np.array([], dtype=np.int64),
        inc_ms.values
    ]))
    inc_padded = df.loc[updated_ms].reset_index()[["start_time","end_time",cols_cancel,cols_trade,col_total,ticker_col,col_total_adj]]
    return state_pad, inc_padded


# ========= 时间/桶工具 =========
def _time_to_bucket_np(t_arr: np.ndarray, interval_ms: int) -> np.ndarray:
    """
    把整数时间 HHMMSSmmm（int64）向下取整到 interval_ms 的桶起点（毫秒数，自午夜起）。
    向量化，无 Python 循环。
    """
    t = t_arr.astype(np.int64)
    h  = t // 10000000
    r1 = t %  10000000
    m  = r1 // 100000
    r2 = r1 %  100000
    s  = r2 // 1000
    ms = r2 %  1000
    total_ms = ((h*3600 + m*60 + s) * 1000) + ms
    return (total_ms // interval_ms) * interval_ms  # 桶起点（毫秒）

def _ms_to_t(ms: np.ndarray) -> np.ndarray:
    """毫秒（自午夜）-> HHMMSSmmm（int64），向量化。"""
    ms = ms % (24*3600*1000)
    h  = ms // (3600*1000)
    r1 = ms %  (3600*1000)
    m  = r1 // (60*1000)
    r2 = r1 %  (60*1000)
    s  = r2 // 1000
    mm = r2 %  1000
    return (h*10000000 + m*100000 + s*1000 + mm).astype(np.int64)


def analyze_deal_cancel_lazy(
    state: Optional[Dict[str, Any]],
    *,
    # —— 必要参数（首轮必须提供）
    idx: Optional[int] = None,
    first_over_60: Optional[int] = None,
    tk: Optional[str] = None,
    interval_ms: int = 10,
    # —— 本步增量（随后的每轮只传这两个）
    deal_df: Optional[pd.DataFrame] = None,           # 本步新增成交
    entrustzt_price_930: Optional[pd.DataFrame] = None,# 本步新增委托（含撤单）
    # —— 列名（按你的数据约定）
    time_col: str = '时间',
    deal_side_col: str = 'BS标志',
    deal_side_sell: str = 'S',
    deal_qty_col: str = '成交数量',
    deal_order_col: str = '叫买序号',
    order_type_col: str = '委托类型',
    order_id_col: str = '交易所委托号',
    order_qty_col: str = '委托数量',
    cancel_code: str = 'D',
    # —— 输出控制
    emit_full: bool = False,                  # True返回全量结果；False仅返回本步新增桶
    adjust_outliers=None,                     # 你的平滑函数；不传则“总量调整=总量”
    window: int = 20,
    nstd: float = 3.0,
):
    """
    懒初始化 + 增量更新版：
      - 首轮：必须提供 idx / first_over_60 / tk（以及首批增量数据，可为空）
      - 后续：只传本步增量 deal_df / entrustzt_price_930
    返回： (state, df_out)
      - df_out：本步新增桶（或全量，取决于 emit_full）
      - state：内部状态，务必在外部保存并传回下一轮
    """
    # ---------- 1) 懒初始化 ----------
    # entrustzt_price_930 = entrustzt_price_930['委托价格'] == 
    # 
    if state is None:
        if idx is None or first_over_60 is None or tk is None:
            raise ValueError("首轮调用必须提供 idx / first_over_60 / tk。")
        state = {
            "idx": int(idx),
            "first_over_60": int(first_over_60),
            "tk": str(tk),
            "interval_ms": int(interval_ms),
            "front_ids": set(),                    # 在我们前面的委托集合
            "last_t": int(first_over_60),          # 已处理到的时间（含阈值）
            "bins_trade": defaultdict(int),        # 桶毫秒 -> 成交总量（卖方 S）
            "bins_cancel": defaultdict(int),       # 桶毫秒 -> 撤单总量（front_ids 的撤单）
            "result": pd.DataFrame(columns=[
                "start_time","end_time","成交总量_撤单","成交总量","总量","总量调整","Ticker"
            ]),
            "timedeal": None,                      # 我们这单的最早成交时刻
        }

    # 没有增量就直接返回
    if (deal_df is None or deal_df.empty) and (entrustzt_price_930 is None or entrustzt_price_930.empty):
        return state, (state["result"].copy() if emit_full else state["result"].iloc[0:0].copy())

    idx   = state["idx"]
    t0    = state["last_t"]
    tk    = state["tk"]
    step_t_candidates = []

    # ---------- 2) 更新“前序委托集合 front_ids” ----------
    # 只要增量里出现了 “first_over_60 之前、且 非撤单、且 委托号<idx” 的行，就加入 front_ids。
    if entrustzt_price_930 is not None and not entrustzt_price_930.empty:
        step_t_candidates.append(int(entrustzt_price_930[time_col].max()))
        o = entrustzt_price_930[[time_col, order_type_col, order_id_col]].copy()
        o[order_id_col] = pd.to_numeric(o[order_id_col], errors='coerce')
        mask_front = (
            (o[time_col] >= state["first_over_60"]) &
            (o[order_type_col] != cancel_code) &
            (o[order_id_col] <= idx)
        )
        if mask_front.any():
            new_ids = o.loc[mask_front, order_id_col].dropna().astype(np.int64).tolist()
            state["front_ids"].update(new_ids)

    # ---------- 3) 记录“我们自己的最早成交时刻” ----------
    if deal_df is not None and not deal_df.empty:
        step_t_candidates.append(int(deal_df[time_col].max()))
        my_rows = deal_df[deal_df[deal_order_col] == idx]
        if not my_rows.empty:
            tmin = int(my_rows[time_col].min())
            if state["timedeal"] is None or tmin < state["timedeal"]:
                state["timedeal"] = tmin

    if not step_t_candidates:
        return state, (state["result"].copy() if emit_full else state["result"].iloc[0:0].copy())
    t1 = max(step_t_candidates)  # 本步处理上限时间

    new_bins = set()
    interval_ms = state["interval_ms"]

    # ---------- 4) 累加“卖方成交 S”到 trade 桶 ----------
    if deal_df is not None and not deal_df.empty:
        d = deal_df[(deal_df[time_col] > t0) & (deal_df[time_col] <= t1)].copy()
        if not d.empty:
            d = d[d[deal_side_col].astype(str).str.upper().eq(str(deal_side_sell).upper())]
            if not d.empty:
                times = d[time_col].to_numpy(np.int64, copy=False)
                vols  = pd.to_numeric(d[deal_qty_col], errors="coerce").fillna(0).astype(np.int64).to_numpy(copy=False)
                buckets = _time_to_bucket_np(times, interval_ms)
                # 累加到 bins_trade
                for b, v in zip(buckets, vols):
                    state["bins_trade"][int(b)] += int(v)
                new_bins.update(int(b) for b in buckets)

    # ---------- 5) 累加"所有撤单 D"到 cancel 桶 ----------
    # 修改：记录所有撤单，而不仅仅是 front_ids 中的撤单
    if entrustzt_price_930 is not None and not entrustzt_price_930.empty:
        o = entrustzt_price_930[(entrustzt_price_930[time_col] >= t0) & (entrustzt_price_930[time_col] <= t1)].copy()
        if not o.empty:
            o = o[o[order_type_col] == cancel_code]
            if not o.empty:
                times = o[time_col].to_numpy(np.int64, copy=False)
                vols  = pd.to_numeric(o[order_qty_col], errors="coerce").fillna(0).astype(np.int64).to_numpy(copy=False)
                buckets = _time_to_bucket_np(times, interval_ms)
                for b, v in zip(buckets, vols):
                    state["bins_cancel"][int(b)] += int(v)
                new_bins.update(int(b) for b in buckets)

    # ---------- 6) 产出新增桶 DataFrame ----------
    if not new_bins:
        state["last_t"] = t1
        return state, (state["result"].copy() if emit_full else state["result"].iloc[0:0].copy())

    bins_sorted = np.array(sorted(new_bins), dtype=np.int64)
    start_times = _ms_to_t(bins_sorted)
    end_times   = _ms_to_t(bins_sorted + interval_ms)

    v_cancel = np.array([state["bins_cancel"].get(int(b), 0) for b in bins_sorted], dtype=np.int64)
    v_trade  = np.array([state["bins_trade"].get(int(b), 0) for b in bins_sorted], dtype=np.int64)
    total    = v_cancel + v_trade

    inc_df = pd.DataFrame({
        "start_time": start_times,
        "end_time":   end_times,
        "成交总量_撤单": v_cancel,
        "成交总量":     v_trade,
        "总量":         total,
        "Ticker":      tk
    }).sort_values("start_time").reset_index(drop=True)

    # ---------- 7) 合并进全量 & （可选）做平滑 ----------
    res = state["result"]
    if res.empty:
        res = inc_df.copy()
    else:
        res = res.set_index("start_time")
        inc = inc_df.set_index("start_time")
        res.update(inc)  # 更新已有桶
        res = pd.concat([res[~res.index.isin(inc.index)], inc], axis=0).sort_index()
        res = res.reset_index()

    if adjust_outliers is not None and not res.empty:
        tail_k = max(window*3, len(inc_df))
        tail = res.iloc[-tail_k:].copy()
        tail = adjust_outliers(tail, col="总量", window=window, nstd=nstd, new_col="总量调整")
        res.loc[tail.index, "总量调整"] = tail["总量调整"].values
        res["总量调整"] = res["总量调整"].fillna(res["总量"])
    else:
        if "总量调整" not in res.columns:
            res["总量调整"] = res["总量"]
        else:
            # 新增部分没有平滑时，先用等值填上
            res.loc[res["总量调整"].isna(), "总量调整"] = res.loc[res["总量调整"].isna(), "总量"]

    # ---------- 8) 写回状态 & 返回 ----------
    state["result"] = res
    state["last_t"] = t1

    out = res.copy() if emit_full else inc_df.assign(总量调整=inc_df["总量"])
    return state, out

# ========= 懒初始化 + 增量更新 =========
# def analyze_deal_cancel_lazy(
#     state: Optional[Dict[str, Any]],
#     *,
#     # —— 必要参数（首轮必须提供）
#     idx: Optional[int] = None,
#     first_over_60: Optional[int] = None,
#     tk: Optional[str] = None,
#     interval_ms: int = 10,
#     # —— 本步增量（随后的每轮只传这两个）
#     deal_df: Optional[pd.DataFrame] = None,           # 本步新增成交
#     entrustzt_price_930: Optional[pd.DataFrame] = None,# 本步新增委托（含撤单）
#     # —— 列名（按你的数据约定）
#     time_col: str = '时间',
#     deal_side_col: str = 'BS标志',
#     deal_side_sell: str = 'S',
#     deal_qty_col: str = '成交数量',
#     deal_order_col: str = '叫买序号',
#     order_type_col: str = '委托类型',
#     order_id_col: str = '交易所委托号',
#     order_qty_col: str = '委托数量',
#     cancel_code: str = 'D',
#     # —— 输出控制
#     emit_full: bool = False,                  # True返回全量结果；False仅返回本步新增桶
#     adjust_outliers=None,                     # 你的平滑函数；不传则“总量调整=总量”
#     window: int = 20,
#     nstd: float = 3.0,
# ):
#     """
#     懒初始化 + 增量更新版：
#       - 首轮：必须提供 idx / first_over_60 / tk（以及首批增量数据，可为空）
#       - 后续：只传本步增量 deal_df / entrustzt_price_930
#     返回： (state, df_out)
#       - df_out：本步新增桶（或全量，取决于 emit_full）
#       - state：内部状态，务必在外部保存并传回下一轮
#     """
#     # ---------- 1) 懒初始化 ----------
#     if state is None:
#         if idx is None or first_over_60 is None or tk is None:
#             raise ValueError("首轮调用必须提供 idx / first_over_60 / tk。")
#         state = {
#             "idx": int(idx),
#             "first_over_60": int(first_over_60),
#             "tk": str(tk),
#             "interval_ms": int(interval_ms),
#             "front_ids": set(),                    # 在我们前面的委托集合
#             "last_t": int(first_over_60),          # 已处理到的时间（含阈值）
#             "bins_trade": defaultdict(int),        # 桶毫秒 -> 成交总量（卖方 S）
#             "bins_cancel": defaultdict(int),       # 桶毫秒 -> 撤单总量（front_ids 的撤单）
#             "result": pd.DataFrame(columns=[
#                 "start_time","end_time","成交总量_撤单","成交总量","总量","总量调整","Ticker"
#             ]),
#             "timedeal": None,                      # 我们这单的最早成交时刻
#         }

#     # 没有增量就直接返回
#     if (deal_df is None or deal_df.empty) and (entrustzt_price_930 is None or entrustzt_price_930.empty):
#         return state, (state["result"].copy() if emit_full else state["result"].iloc[0:0].copy())

#     idx   = state["idx"]
#     t0    = state["last_t"]
#     tk    = state["tk"]
#     step_t_candidates = []

#     # ---------- 2) 更新“前序委托集合 front_ids” ----------
#     # 只要增量里出现了 “first_over_60 之前、且 非撤单、且 委托号<idx” 的行，就加入 front_ids。
#     if entrustzt_price_930 is not None and not entrustzt_price_930.empty:
#         step_t_candidates.append(int(entrustzt_price_930[time_col].max()))
#         o = entrustzt_price_930[[time_col, order_type_col, order_id_col]].copy()
#         o[order_id_col] = pd.to_numeric(o[order_id_col], errors='coerce')
#         mask_front = (
#             (o[time_col] <= state["first_over_60"]) &
#             (o[order_type_col] != cancel_code) &
#             (o[order_id_col] < idx)
#         )
#         if mask_front.any():
#             new_ids = o.loc[mask_front, order_id_col].dropna().astype(np.int64).tolist()
#             state["front_ids"].update(new_ids)

#     # ---------- 3) 记录“我们自己的最早成交时刻” ----------
#     if deal_df is not None and not deal_df.empty:
#         step_t_candidates.append(int(deal_df[time_col].max()))
#         my_rows = deal_df[deal_df[deal_order_col] == idx]
#         if not my_rows.empty:
#             tmin = int(my_rows[time_col].min())
#             if state["timedeal"] is None or tmin < state["timedeal"]:
#                 state["timedeal"] = tmin

#     if not step_t_candidates:
#         return state, (state["result"].copy() if emit_full else state["result"].iloc[0:0].copy())
#     t1 = max(step_t_candidates)  # 本步处理上限时间

#     new_bins = set()
#     interval_ms = state["interval_ms"]

#     # ---------- 4) 累加“卖方成交 S”到 trade 桶 ----------
#     if deal_df is not None and not deal_df.empty:
#         d = deal_df[(deal_df[time_col] > t0) & (deal_df[time_col] <= t1)].copy()
#         if not d.empty:
#             d = d[d[deal_side_col].astype(str).str.upper().eq(str(deal_side_sell).upper())]
#             if not d.empty:
#                 times = d[time_col].to_numpy(np.int64, copy=False)
#                 vols  = pd.to_numeric(d[deal_qty_col], errors="coerce").fillna(0).astype(np.int64).to_numpy(copy=False)
#                 buckets = _time_to_bucket_np(times, interval_ms)
#                 # 累加到 bins_trade
#                 for b, v in zip(buckets, vols):
#                     state["bins_trade"][int(b)] += int(v)
#                 new_bins.update(int(b) for b in buckets)

#     # ---------- 5) 累加“前序撤单 D”到 cancel 桶 ----------
#     if entrustzt_price_930 is not None and not entrustzt_price_930.empty and state["front_ids"]:
#         o = entrustzt_price_930[(entrustzt_price_930[time_col] > t0) & (entrustzt_price_930[time_col] <= t1)].copy()
#         if not o.empty:
#             o = o[o[order_type_col] == cancel_code]
#             if not o.empty:
#                 o[order_id_col] = pd.to_numeric(o[order_id_col], errors='coerce')
#                 o = o[o[order_id_col].isin(state["front_ids"])]
#                 if not o.empty:
#                     times = o[time_col].to_numpy(np.int64, copy=False)
#                     vols  = pd.to_numeric(o[order_qty_col], errors="coerce").fillna(0).astype(np.int64).to_numpy(copy=False)
#                     buckets = _time_to_bucket_np(times, interval_ms)
#                     for b, v in zip(buckets, vols):
#                         state["bins_cancel"][int(b)] += int(v)
#                     new_bins.update(int(b) for b in buckets)

#     # ---------- 6) 产出新增桶 DataFrame ----------
#     if not new_bins:
#         state["last_t"] = t1
#         return state, (state["result"].copy() if emit_full else state["result"].iloc[0:0].copy())

#     bins_sorted = np.array(sorted(new_bins), dtype=np.int64)
#     start_times = _ms_to_t(bins_sorted)
#     end_times   = _ms_to_t(bins_sorted + interval_ms)

#     v_cancel = np.array([state["bins_cancel"].get(int(b), 0) for b in bins_sorted], dtype=np.int64)
#     v_trade  = np.array([state["bins_trade"].get(int(b), 0) for b in bins_sorted], dtype=np.int64)
#     total    = v_cancel + v_trade

#     inc_df = pd.DataFrame({
#         "start_time": start_times,
#         "end_time":   end_times,
#         "成交总量_撤单": v_cancel,
#         "成交总量":     v_trade,
#         "总量":         total,
#         "Ticker":      tk
#     }).sort_values("start_time").reset_index(drop=True)

#     # ---------- 7) 合并进全量 & （可选）做平滑 ----------
#     res = state["result"]
#     if res.empty:
#         res = inc_df.copy()
#     else:
#         res = res.set_index("start_time")
#         inc = inc_df.set_index("start_time")
#         res.update(inc)  # 更新已有桶
#         res = pd.concat([res[~res.index.isin(inc.index)], inc], axis=0).sort_index()
#         res = res.reset_index()

#     if adjust_outliers is not None and not res.empty:
#         tail_k = max(window*3, len(inc_df))
#         tail = res.iloc[-tail_k:].copy()
#         tail = adjust_outliers(tail, col="总量", window=window, nstd=nstd, new_col="总量调整")
#         res.loc[tail.index, "总量调整"] = tail["总量调整"].values
#         res["总量调整"] = res["总量调整"].fillna(res["总量"])
#     else:
#         if "总量调整" not in res.columns:
#             res["总量调整"] = res["总量"]
#         else:
#             # 新增部分没有平滑时，先用等值填上
#             res.loc[res["总量调整"].isna(), "总量调整"] = res.loc[res["总量调整"].isna(), "总量"]

#     # ---------- 8) 写回状态 & 返回 ----------
#     state["result"] = res
#     state["last_t"] = t1

#     out = res.copy() if emit_full else inc_df.assign(总量调整=inc_df["总量"])
#     return state, out

from collections import deque

def update_rolling_mean_simple(state, inc_df, window=100,
                               time_col="start_time", value_col="总量",
                               out_col="总量均值_100"):
    """
    增量滚动均值（简单版）：
      - state: 第一次传 None；函数会返回新的 state，之后重复传回即可
      - inc_df: 本批新增行（建议已补齐 10ms 桶），至少包含 start_time 与 总量
      - 返回: (state, inc_df_with_mean) —— 仅本批行 + 新增均值列
    """
    if inc_df is None or len(inc_df) == 0:
        return state if state is not None else {"buf": deque(), "sum": 0.0, "w": window}, inc_df

    # 懒初始化
    if state is None:
        state = {"buf": deque(), "sum": 0.0, "w": int(window)}

    buf, ssum, w = state["buf"], state["sum"], state["w"]

    # 轻量清洗：时间转 int，按时间排序；总量转 float
    x = inc_df.copy()
    x[time_col] = pd.to_numeric(x[time_col], errors="coerce").astype("Int64")
    x = x.dropna(subset=[time_col]).sort_values(time_col)
    x[time_col] = x[time_col].astype(np.int64)
    x[value_col] = pd.to_numeric(x[value_col], errors="coerce").fillna(0.0).astype(float)

    means = []
    for v in x[value_col].to_numpy():
        buf.append(v)
        ssum += v
        if len(buf) > w:
            ssum -= buf.popleft()
        means.append(ssum / len(buf))   # 不足 w 时，用已有样本均值

    x[out_col] = means

    # 写回状态
    state["buf"], state["sum"] = buf, ssum
    return state, x




def update_rolling_mean_vec(
    state: Optional[Dict[str, Any]],
    inc_df: pd.DataFrame,
    *,
    window: int = 100,
    time_col: str = "start_time",
    value_col: str = "总量",
    out_col: str = "总量均值_100"
):
    """
    增量 + 向量化 的滚动均值（含当前桶；不足窗口用已有样本均值）
    - state: 第一次传 None，会创建 {"w":window, "tail": np.ndarray(dtype=float)}
    - inc_df: 本批新增（建议已按时间升序、10ms 连续补齐），至少含 start_time、总量
    返回: (state, inc_with_mean) —— 仅本批行 + 均值列
    """
    if inc_df is None or inc_df.empty:
        return (state if state is not None else {"w": int(window), "tail": np.empty(0, float)}), inc_df

    # 懒初始化
    if state is None:
        state = {"w": int(window), "tail": np.empty(0, float)}
    w   = int(state["w"])
    tail= state["tail"]

    # 轻量清洗
    x = inc_df.copy()
    x[time_col]  = pd.to_numeric(x[time_col], errors="coerce").astype("Int64")
    x            = x.dropna(subset=[time_col]).sort_values(time_col)
    x[time_col]  = x[time_col].astype(np.int64)
    vals         = pd.to_numeric(x[value_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # 拼接“尾巴 + 新批”，一次向量化计算滚动均值
    arr   = np.concatenate([tail, vals])             # 长度 = len(tail)+len(vals)
    csum  = np.cumsum(arr)
    j     = np.arange(arr.size)

    # 窗口内样本数：前 w-1 逐步累积，之后固定 w
    count = np.minimum(j + 1, w).astype(float)

    # 滚动和：前 w-1 直接用 cumsum，之后用差分
    roll_sum = csum.copy()
    if arr.size > w:
        roll_sum[w:] = csum[w:] - csum[:-w]

    roll_mean = roll_sum / count

    # 只取“本批”的均值（跳过前面的 tail 部分）
    offset = tail.size
    x[out_col] = roll_mean[offset:]

    # 更新 tail：保留末尾 w-1 个值供下一批使用
    if arr.size >= w - 1:
        state["tail"] = arr[-(w - 1):] if w > 1 else np.empty(0, float)
    else:
        state["tail"] = arr.copy()

    return state, x


def update_rolling_mean_simple(
    state: Optional[Dict[str, Any]],
    inc_df: Optional[pd.DataFrame],
    window: int = 100,
    time_col: str = "start_time",
    value_col: str = "总量",
    out_col: str = "总量均值_100",
):
    """
    增量滚动均值（deque 版，含当前；不足窗口用已有样本均值）。
    - state 可为 None 或缺键；函数会自动懒初始化。
    - inc_df 传本批新增（建议已补齐 10ms），至少含 [time_col, value_col]。
    返回: (state, inc_with_mean) —— 仅本批行 + 新均值列
    """
    # —— 懒初始化 / 自愈 —— #
    if (state is None or
        not isinstance(state, dict) or
        any(k not in state for k in ("buf","sum","w")) or
        int(state.get("w", window)) != int(window)):
        state = {"buf": deque(), "sum": 0.0, "w": int(window)}

    if inc_df is None or inc_df.empty:
        return state, (inc_df if inc_df is not None else pd.DataFrame())

    buf = state["buf"]
    ssum = float(state["sum"])
    w    = int(state["w"])

    # 轻量清洗：时间->int，排序去重复；值->float
    x = inc_df.copy()
    x[time_col]  = pd.to_numeric(x[time_col], errors="coerce").astype("Int64")
    x            = x.dropna(subset=[time_col]).sort_values(time_col)
    x            = x[~x.duplicated(subset=[time_col], keep="last")]
    x[time_col]  = x[time_col].astype(np.int64)
    vals         = pd.to_numeric(x[value_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    means = np.empty(len(vals), dtype=float)
    for i in range(len(vals)):
        v = vals[i]
        if len(buf) == w:          # 窗口已满：滑动一步
            ssum += v - buf[0]
            buf.popleft()
            buf.append(v)
        else:                      # 未满：只累加
            buf.append(v)
            ssum += v
        means[i] = ssum / len(buf)

    x[out_col]   = means
    state["sum"] = ssum  # buf 已原位更新
    return state, x


def lookup_from_dict(result: dict, t: int):
    # 确保 key 升序
    keys = sorted(result.keys())
    
    # 遍历找到最后一个 <= t 的 key
    chosen_key = None
    for k in keys:
        if t >= k:
            chosen_key = k
        else:
            break
    
    # 如果 t 比最小的 key 还小，返回 None
    if chosen_key is None:
        return None
    
    return result[chosen_key]
                
#%


def step_limit_buy_with_deals(
    state: Dict[str, Any],
    delta_orders: pd.DataFrame,
    delta_deals: pd.DataFrame,
    *,
    code_col: str = '委托代码',
    order_type_col: str = '委托类型',
    price_col: str = '委托价格',
    order_id_col: str = '交易所委托号',
    qty_col: str = '委托数量',
    time_col: str = '时间',
    buy_code: str = 'B',
    buy_types: Set[str] = frozenset({'A','U','0','1'}),
    cancel_type: str = 'D',
    deal_order_col: str = '叫买序号',
    deal_qty_col: str = '成交数量',
    update_mode: str = 'override',    # 'override' 覆盖新量；'delta' 追加
    order_id_is_int: bool = True,
    emit_empty: bool = False,
    sort_ids: bool = False
) -> pd.DataFrame:
    """
    增量更新涨停价买单队列（挂单/撤单 + 成交扣减），返回“最新队列快照”（通常一行）。
    - 只处理 (state['last_t'], 当前t] 的增量数据
    - 先按委托事件更新队列，再按成交对队列做扣减
    - 队列按 FIFO 维护（OrderedDict），撤单/成交 O(1)

    返回列：['时间','委托价格','总委托数量','委托编号列表','每笔数量列表']
    """
    zt_price = state["zt_price"]
    q: OrderedDict = state["q"]
    total = state["total"]
    last_t = int(state["last_t"])

    # —— 目标时间 t：用两个增量里最大的时间（如果都空，就复用 last_t 并返回空）
    t_candidates = []
    if delta_orders is not None and not delta_orders.empty:
        t_candidates.append(int(delta_orders[time_col].max()))
    if delta_deals is not None and not delta_deals.empty:
        t_candidates.append(int(delta_deals[time_col].max()))
    if not t_candidates:
        return pd.DataFrame(columns=['时间','委托价格','总委托数量','委托编号列表','每笔数量列表'])
    t_now = max(t_candidates)

    # =======================
    # 1) 处理“委托增量” (A/U/0/1/D)
    # =======================
    if delta_orders is not None and not delta_orders.empty:
        df = delta_orders[[time_col, order_type_col, order_id_col, qty_col, price_col, code_col]].copy()

        # 基本清洗与筛选（只处理买、涨停价、有效类型、且时间在 (last_t, t_now] ）
        df[code_col] = df[code_col].astype(str).str.strip().str.upper()
        df[order_type_col] = df[order_type_col].astype(str).str.strip().str.upper()
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce')
        df = df[(df[code_col] == buy_code)
                & (df[price_col] == float(zt_price))
                & (df[order_type_col].isin(buy_types | {cancel_type}))
                & (df[time_col] > last_t) & (df[time_col] <= t_now)]
        if not df.empty:
            # 排序，保证同一时刻的处理顺序稳定
            df = df.sort_values([time_col, order_id_col], ignore_index=True)

            times = df[time_col].to_numpy(np.int64, copy=False)
            types = df[order_type_col].to_numpy(dtype='U1', copy=False)
            ids   = (df[order_id_col].to_numpy(np.int64, copy=False)
                     if order_id_is_int else df[order_id_col].astype(str).to_numpy(copy=False))
            qtys  = df[qty_col].to_numpy(np.int64, copy=False)

            btypes = buy_types
            add_is_delta = (update_mode.lower() == 'delta')

            for i in range(len(df)):
                tp  = types[i]
                oid = ids[i]
                qv  = int(qtys[i])
                if tp in btypes:  # 新挂单/改单
                    prev = q.get(oid)
                    if prev is None:
                        q[oid] = qv
                        total += qv
                    else:
                        if add_is_delta:
                            q[oid] = prev + qv
                            total += qv
                        else:  # override
                            if qv != prev:
                                total += (qv - prev)
                                q[oid] = qv
                elif tp == cancel_type:
                    prev = q.pop(oid, None)
                    if prev is not None:
                        total -= prev

    # =======================
    # 2) 处理“成交增量” (对队列扣减)
    # =======================
    if delta_deals is not None and not delta_deals.empty:
        d = delta_deals[[time_col, deal_order_col, deal_qty_col]].copy()
        d[deal_qty_col] = pd.to_numeric(d[deal_qty_col], errors='coerce')
        d = d[(d[time_col] > last_t) & (d[time_col] <= t_now)]
        if not d.empty:
            # 只需要按 order_id 聚合成交量即可（一个时刻可能多笔同一单）
            d[deal_order_col] = d[deal_order_col].astype(str)
            agg = (d.groupby(deal_order_col, as_index=False)[deal_qty_col]
                    .sum()
                    .rename(columns={deal_qty_col: '成交总量'}))
            # 扣减：只对当前队列中存在的委托号生效
            for _, r in agg.iterrows():
                oid_str = r[deal_order_col]
                oid = int(float(oid_str)) if order_id_is_int else oid_str
                dec = int(r['成交总量'])
                if dec <= 0:
                    continue
                prev = q.get(oid)
                if prev is None:
                    continue  # 成交里出现了不在队列的委托号，直接忽略
                if dec >= prev:
                    total -= prev
                    q.pop(oid, None)
                else:
                    q[oid] = prev - dec
                    total -= dec

    # =======================
    # 3) 产出“当前时刻 t_now”的快照
    # =======================
    if (total <= 0) and (not emit_empty):
        # 队列空且不要求产出空快照
        state["total"]  = total
        state["last_t"] = t_now
        return pd.DataFrame(columns=['时间','委托价格','总委托数量','委托编号列表','每笔数量列表'])

    items = sorted(q.items(), key=lambda kv: kv[0]) if sort_ids else q.items()
    snap = pd.DataFrame([{
        '时间': int(t_now),
        '委托价格': float(zt_price),
        '总委托数量': int(total),
        '委托编号列表': [int(k) if order_id_is_int else str(k) for k,_ in items],
        '每笔数量列表': [int(v) for _,v in items]
    }])

    # 写回状态并返回
    state["total"]  = total
    state["last_t"] = t_now
    return snap


def init_limit_buy_state(zt_price: Union[int, float]):
    return {
        "zt_price": float(zt_price),
        "q": OrderedDict(),  # order_id -> qty (FIFO)
        "total": 0,
        "last_t": 0,
    }

def update_limit_buy_state_incremental(
    state: Dict[str, Any],
    delta_df: pd.DataFrame,
    code_col: str = '委托代码',
    order_type_col: str = '委托类型',
    price_col: str = '委托价格',
    order_id_col: str = '交易所委托号',
    qty_col: str = '委托数量',
    time_col: str = '时间',
    buy_code: str = 'B',
    buy_types: Set[str] = frozenset({'A','U','0','1'}),
    cancel_type: str = 'D',
    emit_each_event: bool = False,   # 一般 false：每个时刻只产出一次
    sort_ids: bool = False,          # 是否按编号升序输出
    order_id_is_int: bool = True     # 如果列已是 int，设 True 会更快
) -> pd.DataFrame:
    """只处理增量 delta_df，返回本批产生的快照（通常就是当前 t 一条）"""
    if delta_df is None or delta_df.empty:
        return pd.DataFrame(columns=['时间','委托价格','总委托数量','委托编号列表','每笔数量列表'])

    zt_price = state["zt_price"]
    q: OrderedDict = state["q"]
    total = state["total"]

    # —— 预筛：只处理（B、涨停价、有效类型）
    df = delta_df[[time_col, order_type_col, order_id_col, qty_col, price_col, code_col]].copy()
    m = (
        (df[code_col] == buy_code) &
        (df[price_col] == zt_price) &
        (df[order_type_col].isin(buy_types | {cancel_type}))
    )
    df = df.loc[m]
    if df.empty:
        return pd.DataFrame(columns=['时间','委托价格','总委托数量','委托编号列表','每笔数量列表'])

    # —— 排序（增量很小，随便选）
    df = df.sort_values([time_col, order_id_col], ignore_index=True)

    # —— 转数组（增量通常极小，开销非常低）
    times = df[time_col].to_numpy(np.int64, copy=False)
    types = df[order_type_col].astype('U1').to_numpy(copy=False)
    ids = df[order_id_col].to_numpy(np.int64, copy=False) if order_id_is_int else df[order_id_col].astype(str).to_numpy(copy=False)
    qtys = df[qty_col].to_numpy(np.int64, copy=False)

    out_rows = []
    n = len(df)
    btypes = buy_types

    def emit_snapshot(t_emit: int):
        nonlocal out_rows, total
        if total <= 0:
            return
        items = sorted(q.items(), key=lambda kv: kv[0]) if sort_ids else q.items()
        out_rows.append({
            '时间': int(t_emit),
            '委托价格': float(zt_price),
            '总委托数量': int(total),
            '委托编号列表': [int(k) if order_id_is_int else str(k) for k,_ in items],
            '每笔数量列表': [int(v) for _,v in items],
        })

    for i in range(n):
        t = int(times[i])
        tp = types[i]
        oid = ids[i]
        qv  = int(qtys[i])

        if tp in btypes:     # 新挂单/改单
            prev = q.get(oid)
            if prev is None:
                q[oid] = qv
                total += qv
            else:
                if qv != prev:   # 若“U”是追加而非覆盖，请改为：q[oid]=prev+qv；total+=qv
                    total += (qv - prev)
                    q[oid] = qv
        elif tp == cancel_type:
            prev = q.pop(oid, None)
            if prev is not None:
                total -= prev

        if emit_each_event:
            emit_snapshot(t)
        else:
            if i+1 == n or times[i+1] != t:  # 时刻边界
                emit_snapshot(t)

    # 写回状态
    state["total"] = total
    state["last_t"] = int(times[-1])
    return pd.DataFrame(out_rows)


# def gen_timestamps(start: int, end: int = 145700000, step: int = 10):
#     """
#     生成从 start 到 end 的时间戳序列，每隔 step 毫秒一个。
    
#     参数:
#         start : int   起始时间戳，例如 145646480 -> 14:56:46.480
#         end   : int   结束时间戳，默认 150000000 -> 15:00:00.000
#         step  : int   步长，毫秒数，默认 10
    
#     返回:
#         List[int] 时间戳序列
#     """
#     if start > end:
#         raise ValueError("起始时间必须小于结束时间")
#     return list(range(start, end + 1, step))


def gen_timestamps(start: int, end: int = 150000000, step: int = 10):
    """
    生成从 start 到 end 的时间戳序列，只包含股票交易时间。
    输入和输出的时间戳格式为 HHMMSSmmm（9位整数）。
    """
    def time_to_ms(t: int) -> int:
        str_t = str(t).zfill(9)
        hour = int(str_t[0:2])
        minute = int(str_t[2:4])
        second = int(str_t[4:6])
        millisecond = int(str_t[6:9])
        return (hour * 3600 + minute * 60 + second) * 1000 + millisecond
    
    def ms_to_time(ms: int) -> int:
        total_ms = ms % 1000
        total_seconds = ms // 1000
        hour = total_seconds // 3600
        minute = (total_seconds % 3600) // 60
        second = total_seconds % 60
        return hour * 10000000 + minute * 100000 + second * 1000 + total_ms
    
    if start > end:
        raise ValueError("起始时间必须小于结束时间")
    
    start_ms = time_to_ms(start)
    end_ms = time_to_ms(end)
    
    # 交易时间段（毫秒）
    trading_periods = [
        (time_to_ms(93000000), time_to_ms(113000000)),  # 9:30-11:30
        (time_to_ms(130000000), time_to_ms(150000000))  # 13:00-15:00
    ]
    
    result = []
    for period_start, period_end in trading_periods:
        segment_start = max(start_ms, period_start)
        segment_end = min(end_ms, period_end)
        
        if segment_start <= segment_end:
            # 计算该段内的所有时间戳
            num_steps = (segment_end - segment_start) // step + 1
            result.extend([
                ms_to_time(segment_start + i * step)
                for i in range(num_steps)
                if segment_start + i * step <= segment_end
            ])
    
    return result


from collections import OrderedDict
from typing import List, Dict, Any, Set, Union
import numpy as np
import pandas as pd

def track_buy_queue_at_limit_arrayfast(
    df: pd.DataFrame,
    zt_price: Union[int, float],
    code_col: str = '委托代码',
    order_type_col: str = '委托类型',
    price_col: str = '委托价格',
    order_id_col: str = '交易所委托号',
    qty_col: str = '委托数量',
    time_col: str = '时间',
    buy_code: str = 'B',
    buy_types: Set[str] = frozenset({'A','U','0','1'}),   # 按你的市场约定
    cancel_type: str = 'D',
    emit_each_event: bool = False,    # True=每条事件都快照；False=每个时间戳出一次（更快）
    sort_ids: bool = False,           # True=按编号升序输出；False=按队列FIFO顺序输出
    cast_order_id_to_int: bool = True # True=委托号转int，性能和内存更好
) -> pd.DataFrame:
    """数组版高速队列跟踪，返回列：
       ['时间','委托价格','总委托数量','委托编号列表','每笔数量列表']"""
    if df.empty:
        return pd.DataFrame(columns=['时间','委托价格','总委托数量','委托编号列表','每笔数量列表'])

    # —— 预筛（买方向 + 涨停价 + 有效类型）
    # 先尽量转为基础类型，避免对象列
    df_local = df[[time_col, order_type_col, order_id_col, qty_col, price_col, code_col]].copy()
    # 精确比较建议用整数价位（例如 price_tick_idx），若现在是浮点且来自同源生成，也可直接 ==
    mask = (
        (df_local[code_col] == buy_code) &
        (df_local[price_col] == zt_price) &
        (df_local[order_type_col].isin(buy_types | {cancel_type}))
    )
    df_local = df_local.loc[mask]
    if df_local.empty:
        return pd.DataFrame(columns=['时间','委托价格','总委托数量','委托编号列表','每笔数量列表'])

    # —— 排序（稳定）
    df_local = df_local.sort_values([time_col, order_id_col], kind='mergesort', ignore_index=True)

    # —— 列转 NumPy（尽量用基础 dtype）
    times = df_local[time_col].to_numpy(np.int64, copy=False)
    types = df_local[order_type_col].astype('U1').to_numpy(copy=False)  # 单字符更省
    if cast_order_id_to_int:
        ids = pd.to_numeric(df_local[order_id_col], errors='coerce').fillna(-1).astype(np.int64).to_numpy(copy=False)
    else:
        ids = df_local[order_id_col].astype(str).to_numpy(copy=False)
    qtys = df_local[qty_col].to_numpy(np.int64, copy=False)
    price_val = float(zt_price)

    # —— 队列容器与输出缓存
    q = OrderedDict()   # order_id -> qty
    total_qty = 0

    out_t, out_p, out_ids, out_qtys, out_tot = [], [], [], [], []
    n = len(times)
    btypes = buy_types  # 局部引用加速属性查找

    def emit_snapshot(t_emit: int):
        if total_qty <= 0:
            return
        if sort_ids:
            items = sorted(q.items(), key=lambda kv: kv[0])
        else:
            items = q.items()
        out_t.append(int(t_emit))
        out_p.append(price_val)
        out_ids.append([int(k) if cast_order_id_to_int else str(k) for k, _ in items])
        out_qtys.append([int(v) for _, v in items])
        out_tot.append(int(total_qty))

    # —— 主循环（顺序扫描）
    for i in range(n):
        t = int(times[i])
        tp = types[i]
        oid = ids[i]
        qv  = int(qtys[i])

        if tp in btypes:
            prev = q.get(oid)
            if prev is None:
                q[oid] = qv
                total_qty += qv
            else:
                # 如 U 表示覆盖新量（重报），这里覆盖；如表示追加，把下面两行换成：q[oid]=prev+qv; total_qty+=qv
                if qv != prev:
                    total_qty += (qv - prev)
                    q[oid] = qv
        elif tp == cancel_type:
            prev = q.pop(oid, None)
            if prev is not None:
                total_qty -= prev

        # —— 产出时机
        if emit_each_event:
            emit_snapshot(t)
        else:
            if i+1 == n or times[i+1] != t:
                emit_snapshot(t)

    return pd.DataFrame({
        '时间': out_t,
        '委托价格': out_p,
        '总委托数量': out_tot,
        '委托编号列表': out_ids,
        '每笔数量列表': out_qtys,
    })



def log_time(func_name, t_mark, ms, extra=None):
    rec = {"func": func_name, "t": int(t_mark), "ms": float(ms)}
    if extra:
        rec.update(extra)
    TIMINGS.append(rec)


# ---------- 时间工具（HHMMSSmmm <-> 毫秒，自午夜起） ----------


def _ms_to_t_vec(ms: np.ndarray) -> np.ndarray:
    ms = ms % (24*3600*1000)
    h  = ms // (3600*1000)
    r1 = ms %  (3600*1000)
    m  = r1 // (60*1000)
    r2 = r1 %  (60*1000)
    s  = r2 // 1000
    mm = r2 %  1000
    return (h*10000000 + m*100000 + s*1000 + mm).astype(np.int64)

# ---------- 懒初始化 ----------
def init_forecast_state(window:int=1000, diff_window:int=1000, phi:float=0.5,
                        interval_ms:int=10) -> Dict[str, Any]:
    return {
        "W": int(window),                  # 水平窗口
        "D": int(diff_window),             # 斜率窗口（对一阶差分）
        "phi": float(phi),                 # 阻尼系数 (0<phi<=1)
        "interval_ms": int(interval_ms),   # 桶间隔 (10ms)
        "buf": deque(),                    # 最近 W 个值
        "sum": 0.0,                        # buf 的和
        "diffs": deque(),                  # 最近 D 个一阶差分
        "sum_diff": 0.0,                   # diffs 的和
        "last_time": None,                 # 最近一个 start_time (HHMMSSmmm)
        "ticker": None                     # 最近 Ticker（可选）
    }


# state = fc_state inc_df = inc_padded
# 反推预测下一个10ms的成交
# def update_and_predict_next10ms_with_carry(
#     state: Optional[Dict[str, Any]],
#     inc_df: Optional[pd.DataFrame],
#     *,
#     ahead_total: float,                  # 👈 新增：未来 1s / 3min 总量
#     horizon_ms: int = 1000,
#     time_col: str = "start_time",
#     value_col: str = "总量",
#     ticker_col: str = "Ticker",
# ):
#     """
#     增量更新 + 反演约束预测
#     返回：state, pred_next_10ms
#     """

#     # ---------- 初始化 ----------
#     if state is None:
#         state = init_forecast_state()

#     interval = state["interval_ms"]
#     phi      = state["phi"]
#     W, D     = state["W"], state["D"]

#     # ---------- 增量更新（和你原函数一完全一致） ----------
#     if inc_df is not None and not inc_df.empty:
#         x = inc_df.sort_values(time_col)
#         vals = pd.to_numeric(x[value_col], errors="coerce").fillna(0.0).to_numpy()

#         buf, ssum = state["buf"], state["sum"]
#         diffs, sdiff = state["diffs"], state["sum_diff"]

#         prev = buf[-1] if len(buf) else None
#         for v in vals:
#             if prev is not None:
#                 d = v - prev
#                 diffs.append(d)
#                 sdiff += d
#                 if len(diffs) > D:
#                     sdiff -= diffs.popleft()
#             prev = v

#             buf.append(v)
#             ssum += v
#             if len(buf) > W:
#                 ssum -= buf.popleft()

#         state["sum"] = ssum
#         state["sum_diff"] = sdiff
#         state["last_time"] = int(x[time_col].iloc[-1])

#     # ---------- 估计 slope ----------
#     slope = state["sum_diff"] / len(state["diffs"]) if state["diffs"] else 0.0

#     # ---------- Holt 阻尼项 ----------
#     H = int(horizon_ms // interval)

#     def g(k):
#         return (1 - phi**k) / (1 - phi) if phi != 1.0 else float(k)

#     sum_g = sum(g(k) for k in range(1, H + 1))
#     g1 = g(1)

#     # ---------- 反演 level* ----------
#     level_star = max(0.0, (ahead_total - slope * sum_g) / H)

#     # ---------- 预测下一 10ms ----------
#     pred_next_10ms = max(0.0, level_star + slope * g1)

#     return state, pred_next_10ms

# def update_and_predict_next10ms_with_carry(
#     state: Optional[Dict[str, Any]],
#     inc_df: Optional[pd.DataFrame],
#     *,
#     ahead_total: Union[float, List[float], np.ndarray, pd.Series],  # 👈 支持单个值或数组/Series
#     horizon_ms: int = 1000,
#     time_col: str = "start_time",
#     value_col: str = "总量",
#     ticker_col: str = "Ticker",
# ):
#     """
#     增量更新 + 反演约束预测
#     返回：state, pred_next_10ms (单个值或数组/Series)
#     """

#     # ---------- 初始化 ----------
#     if state is None:
#         state = init_forecast_state()

#     interval = state["interval_ms"]
#     phi      = state["phi"]
#     W, D     = state["W"], state["D"]

#     # ---------- 增量更新（和你原函数一完全一致） ----------
#     if inc_df is not None and not inc_df.empty:
#         x = inc_df.sort_values(time_col)
#         vals = pd.to_numeric(x[value_col], errors="coerce").fillna(0.0).to_numpy()

#         buf, ssum = state["buf"], state["sum"]
#         diffs, sdiff = state["diffs"], state["sum_diff"]

#         prev = buf[-1] if len(buf) else None
#         for v in vals:
#             if prev is not None:
#                 d = v - prev
#                 diffs.append(d)
#                 sdiff += d
#                 if len(diffs) > D:
#                     sdiff -= diffs.popleft()
#             prev = v

#             buf.append(v)
#             ssum += v
#             if len(buf) > W:
#                 ssum -= buf.popleft()

#         state["sum"] = ssum
#         state["sum_diff"] = sdiff
#         state["last_time"] = int(x[time_col].iloc[-1])

#     # ---------- 估计 slope ----------
#     slope = state["sum_diff"] / len(state["diffs"]) if state["diffs"] else 0.0

#     # ---------- Holt 阻尼项 ----------
#     H = int(horizon_ms // interval)

#     def g(k):
#         return (1 - phi**k) / (1 - phi) if phi != 1.0 else float(k)

#     sum_g = sum(g(k) for k in range(1, H + 1))
#     g1 = g(1)

#     # ---------- 转换为 numpy array ----------
#     if isinstance(ahead_total, pd.Series):
#         ahead_total_arr = ahead_total.dropna().values
#         is_series = True
#         series_index = ahead_total.dropna().index
#     elif isinstance(ahead_total, (list, np.ndarray)):
#         ahead_total_arr = np.asarray(ahead_total)
#         ahead_total_arr = ahead_total_arr[~np.isnan(ahead_total_arr)] if ahead_total_arr.size > 0 else ahead_total_arr
#         is_series = False
#         series_index = None
#     else:
#         # 单个值
#         ahead_total_arr = np.array([ahead_total])
#         is_series = False
#         series_index = None

#     # ---------- 向量化计算 level* 和 pred_next_10ms ----------
#     level_star = np.maximum(0.0, (ahead_total_arr - slope * sum_g) / H)
#     pred_next_10ms_arr = np.maximum(0.0, level_star + slope * g1)

#     # ---------- 返回格式 ----------
#     if is_series:
#         pred_next_10ms = pd.Series(pred_next_10ms_arr, index=series_index)
#     elif len(pred_next_10ms_arr) == 1 and not isinstance(ahead_total, (list, np.ndarray, pd.Series)):
#         # 如果输入是单个值，返回单个值（向后兼容）
#         pred_next_10ms = float(pred_next_10ms_arr[0])
#     else:
#         # 返回 numpy array
#         pred_next_10ms = pred_next_10ms_arr

#     return state, pred_next_10ms
# inc_df = inc_padded
# state = fc_state
def update_and_predict_next10ms_with_carryV1(
    state: Optional[Dict[str, Any]],
    inc_df: Optional[pd.DataFrame],
    *,
    ahead_total: Union[float, List[float], np.ndarray, pd.Series],
    index: Optional[pd.Series] = None,
    horizon_ms: int = 1000,
    time_col: str = "start_time",
    value_col: str = "总量",
    ticker_col: str = "Ticker",
    # ===== 新增控制参数 =====
    alpha: float = 0.35,     # slope 保守折扣
    p_min: float = 0.05,     # 最低成交概率（防止完全归零）
    use_nonlinear: bool = True,  # 是否启用非线性压缩
):
    """
    增量更新 + 反演约束预测 + 成交概率 gating（10ms 稳定版）
    返回：state, pred_next_10ms, slope_eff, level_star, p_trade
    """

    # ---------- 初始化 ----------
    if state is None:
        state = init_forecast_state()

    interval = state["interval_ms"]
    phi      = state["phi"]
    W, D     = state["W"], state["D"]

    # ---------- 增量更新 ----------
    if inc_df is not None and not inc_df.empty:
        x = inc_df.sort_values(time_col)
        vals = pd.to_numeric(x[value_col], errors="coerce").fillna(0.0).to_numpy()

        buf, ssum = state["buf"], state["sum"]
        diffs, sdiff = state["diffs"], state["sum_diff"]

        prev = buf[-1] if len(buf) else None
        for v in vals:
            if prev is not None:
                d = v - prev
                diffs.append(d)
                sdiff += d
                if len(diffs) > D:
                    sdiff -= diffs.popleft()
            prev = v

            buf.append(v)
            ssum += v
            if len(buf) > W:
                ssum -= buf.popleft()

        state["sum"] = ssum
        state["sum_diff"] = sdiff
        state["last_time"] = int(x[time_col].iloc[-1])

    # ---------- slope（保守版） ----------
    raw_slope = state["sum_diff"] / len(state["diffs"]) if state["diffs"] else 0.0
    slope_eff = raw_slope * alpha

    # ---------- Holt 阻尼 ----------
    H = int(horizon_ms // interval)

    def g(k):
        return (1 - phi**k) / (1 - phi) if phi != 1.0 else float(k)

    sum_g = sum(g(k) for k in range(1, H + 1))
    g1 = g(1)

    # ---------- ahead_total → ndarray ----------
    if isinstance(ahead_total, pd.Series):
        valid_mask = ~ahead_total.isna()
        ahead_arr = ahead_total[valid_mask].values
        series_index = ahead_total[valid_mask].index
        is_series = True
    elif isinstance(ahead_total, (list, np.ndarray)):
        ahead_arr = np.asarray(ahead_total)
        ahead_arr = ahead_arr[~np.isnan(ahead_arr)] if ahead_arr.size else ahead_arr
        series_index = index if index is not None and len(index) == len(ahead_arr) else None
        is_series = False
    else:
        ahead_arr = np.array([ahead_total])
        series_index = None
        is_series = False

    # ---------- level* ----------
    level_star = np.maximum(
        0.0,
        (ahead_arr - slope_eff * sum_g) / max(H, 1)
    )

    raw_pred = np.maximum(0.0, level_star + slope_eff * g1)

    # ---------- 非线性压缩（强烈推荐） ----------
    if use_nonlinear:
        raw_pred = np.sqrt(raw_pred)

    # ---------- 成交发生概率 gating（核心） ----------
    trade_event_rate = len(state["diffs"]) / max(W, 1)
    p_trade = min(1.0, max(p_min, trade_event_rate))

    pred_next_10ms_arr = raw_pred * p_trade

    # ---------- 返回格式 ----------
    if is_series and series_index is not None:
        pred_next_10ms = pd.Series(pred_next_10ms_arr, index=series_index)
    elif len(pred_next_10ms_arr) == 1 and not isinstance(ahead_total, (list, np.ndarray, pd.Series)):
        pred_next_10ms = float(pred_next_10ms_arr[0])
    else:
        pred_next_10ms = pred_next_10ms_arr

    return state, pred_next_10ms, slope_eff, level_star, p_trade



def update_and_predict_next100ms_with_carry(
    state: Optional[Dict[str, Any]],
    inc_df: Optional[pd.DataFrame],
    *,
    ahead_total: Union[float, List[float], np.ndarray, pd.Series],
    index: Optional[pd.Series] = None,
    horizon_ms: int = 1000,
    time_col: str = "start_time",
    value_col: str = "总量",
    alpha: float = 0.7,          # 100ms 下可更激进
    use_nonlinear: bool = False  # 100ms 可关闭
):
    """
    100ms 粒度：
    增量更新 + Holt carry + 反演约束预测
    返回未来一个 100ms 的期望消耗
    """

    # ---------- 初始化 ----------
    if state is None:
        state = init_forecast_state(interval_ms=100)

    interval = state["interval_ms"]   # =100
    phi      = state["phi"]
    W, D     = state["W"], state["D"]

    # ---------- 增量更新 ----------
    if inc_df is not None and not inc_df.empty:
        x = inc_df.sort_values(time_col)
        vals = pd.to_numeric(x[value_col], errors="coerce").fillna(0.0).to_numpy()

        buf, ssum = state["buf"], state["sum"]
        diffs, sdiff = state["diffs"], state["sum_diff"]

        prev = buf[-1] if len(buf) else None
        for v in vals:
            if prev is not None:
                d = v - prev
                diffs.append(d)
                sdiff += d
                if len(diffs) > D:
                    sdiff -= diffs.popleft()
            prev = v

            buf.append(v)
            ssum += v
            if len(buf) > W:
                ssum -= buf.popleft()

        state["sum"] = ssum
        state["sum_diff"] = sdiff
        state["last_time"] = int(x[time_col].iloc[-1])

    # ---------- slope ----------
    slope = state["sum_diff"] / len(state["diffs"]) if state["diffs"] else 0.0
    slope_eff = slope * alpha

    # ---------- Holt carry ----------
    H = int(horizon_ms // interval)

    def g(k):
        return (1 - phi**k) / (1 - phi) if phi != 1.0 else float(k)

    sum_g = sum(g(k) for k in range(1, H + 1))
    g1 = g(1)

    # ---------- ahead_total → ndarray ----------
    if isinstance(ahead_total, pd.Series):
        valid = ~ahead_total.isna()
        ahead_arr = ahead_total[valid].values
        series_index = ahead_total[valid].index
        is_series = True
    elif isinstance(ahead_total, (list, np.ndarray)):
        ahead_arr = np.asarray(ahead_total)
        series_index = index if index is not None else None
        is_series = False
    else:
        ahead_arr = np.array([ahead_total])
        series_index = None
        is_series = False

    # ---------- level* ----------
    level_star = np.maximum(
        0.0,
        (ahead_arr - slope_eff * sum_g) / max(H, 1)
    )

    pred_100ms = np.maximum(0.0, level_star + slope_eff * g1)

    # ---------- 可选非线性 ----------
    if use_nonlinear:
        pred_100ms = np.sqrt(pred_100ms)

    # ---------- 返回 ----------
    if is_series and series_index is not None:
        pred = pd.Series(pred_100ms, index=series_index)
    elif len(pred_100ms) == 1:
        pred = float(pred_100ms[0])
    else:
        pred = pred_100ms

    return state, pred, slope_eff, level_star




  # ahead_total_series = result_df['排队数量'].dropna()
  #                       fc_state, pred_next,slope,level_star = update_and_predict_next10ms_with_carry( fc_state,
  #                                                                               inc_df=inc_padded,  
  #                                                                               ahead_total = ahead_total_series,
  #                                                                               index = result_df['交易所委托号'].dropna(),
  #                                                                               horizon_ms=1000,        
  #                                                                               time_col="start_time",
  #                                                                               value_col="总量",
  #                                                                               ticker_col="Ticker"
  #                                                                           )
                        
                        
def update_and_predict_next10ms_with_carry(
    state: Optional[Dict[str, Any]],
    inc_df: Optional[pd.DataFrame],
    *,
    ahead_total: Union[float, List[float], np.ndarray, pd.Series],  # 👈 支持单个值或数组/Series
    index: Optional[pd.Series] = None,  # 👈 可选的索引，如果提供则用作返回值的索引
    horizon_ms: int = 1000,
    time_col: str = "start_time",
    value_col: str = "总量",
    ticker_col: str = "Ticker",
):
    """
    增量更新 + 反演约束预测
    返回：state, pred_next_10ms (单个值或数组/Series)
    """

    # ---------- 初始化 ----------
    if state is None:
        state = init_forecast_state()

    interval = state["interval_ms"]
    phi      = state["phi"]=0.5
    W, D     = state["W"], state["D"]

    # ---------- 增量更新（和你原函数一完全一致） ----------
    if inc_df is not None and not inc_df.empty:
        x = inc_df.sort_values(time_col)
        vals = pd.to_numeric(x[value_col], errors="coerce").fillna(0.0).to_numpy()

        buf, ssum = state["buf"], state["sum"]
        diffs, sdiff = state["diffs"], state["sum_diff"]

        prev = buf[-1] if len(buf) else None
        for v in vals:
            if prev is not None:
                d = v - prev
                diffs.append(d)
                sdiff += d
                if len(diffs) > D:
                    sdiff -= diffs.popleft()
            prev = v

            buf.append(v)
            ssum += v
            if len(buf) > W:
                ssum -= buf.popleft()

        state["sum"] = ssum
        state["sum_diff"] = sdiff
        state["last_time"] = int(x[time_col].iloc[-1])

    # ---------- 估计 slope ----------
    slope = state["sum_diff"] / len(state["diffs"]) if state["diffs"] else 0.0

    # ---------- Holt 阻尼项 ----------
    H = int(horizon_ms // interval)

    def g(k):
        return (1 - phi**k) / (1 - phi) if phi != 1.0 else float(k)

# ag = [g(k) for k in range(1, H + 1)]

    sum_g = sum(g(k) for k in range(1, H + 1))
    g1 = g(1)

    # ---------- 转换为 numpy array ----------
    if isinstance(ahead_total, pd.Series):
        # 保存原始索引，用于后续匹配
        valid_mask = ~ahead_total.isna()
        ahead_total_arr = ahead_total[valid_mask].values
        is_series = True
        # 如果提供了 index 参数，使用它；否则使用 ahead_total 的索引
        if index is not None and isinstance(index, pd.Series):
            # 确保 index 的长度与过滤后的 ahead_total 一致
            if len(index) == len(ahead_total):
                series_index = index[valid_mask]
            elif len(index) == len(ahead_total_arr):
                series_index = index
            else:
                # 如果长度不匹配，使用 ahead_total 的索引
                series_index = ahead_total[valid_mask].index
        else:
            series_index = ahead_total[valid_mask].index
    elif isinstance(ahead_total, (list, np.ndarray)):
        ahead_total_arr = np.asarray(ahead_total)
        ahead_total_arr = ahead_total_arr[~np.isnan(ahead_total_arr)] if ahead_total_arr.size > 0 else ahead_total_arr
        is_series = False
        series_index = index if index is not None and len(index) == len(ahead_total_arr) else None
    else:
        # 单个值
        ahead_total_arr = np.array([ahead_total])
        is_series = False
        series_index = None

    # ---------- 向量化计算 level* 和 pred_next_10ms ----------
    level_star = np.maximum(0.0, (ahead_total_arr - slope * sum_g) / H)
    pred_next_10ms_arr = np.maximum(0.0, level_star + slope * g1)

    # ---------- 返回格式 ----------
    if is_series or (series_index is not None and len(pred_next_10ms_arr) > 0):
        pred_next_10ms = pd.Series(pred_next_10ms_arr, index=series_index)
    elif len(pred_next_10ms_arr) == 1 and not isinstance(ahead_total, (list, np.ndarray, pd.Series)):
        # 如果输入是单个值，返回单个值（向后兼容）
        pred_next_10ms = float(pred_next_10ms_arr[0])
    else:
        # 返回 numpy array
        pred_next_10ms = pred_next_10ms_arr

    return state, pred_next_10ms,slope,level_star


# ---------- 增量更新 + 预测（未来 3 分钟） ----------
def update_and_forecast_10ms(
    state: Optional[Dict[str, Any]],
    inc_df: Optional[pd.DataFrame],
    *,
    horizon_ms: int = 180_000,            # 3 分钟
    time_col: str = "start_time",
    value_col: str = "总量",
    ticker_col: str = "Ticker",
) -> (Dict[str, Any], pd.DataFrame):
    """
    传入“本批新增的 10ms 桶”（至少含 start_time、总量），返回未来3分钟逐桶预测。
    预测法：level=最近W均值；trend=最近D个差分均值；阻尼 Holt-like: g(k)=(1-phi^k)/(1-phi)
    """
    # 懒初始化 / 自愈
    if (state is None) or (not isinstance(state, dict)) or any(k not in state for k in
        ("W","D","phi","interval_ms","buf","sum","diffs","sum_diff","last_time","ticker")):
        state = init_forecast_state()

    # 空增量，直接按当前 state 预测
    if inc_df is None or inc_df.empty:
        return state, _forecast_from_state(state, horizon_ms, time_col, ticker_col)

    # —— 轻量清洗：时间升序、去重，总量转 float —— #
    x = inc_df.copy()
    x[time_col]  = pd.to_numeric(x[time_col], errors="coerce").astype("Int64")
    x            = x.dropna(subset=[time_col]).sort_values(time_col)
    x            = x[~x.duplicated(subset=[time_col], keep="last")]
    x[time_col]  = x[time_col].astype(np.int64)
    vals         = pd.to_numeric(x[value_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # —— 增量更新 level 和 trend —— #
    W, D = state["W"], state["D"]
    buf, ssum = state["buf"], float(state["sum"])
    diffs, sdiff = state["diffs"], float(state["sum_diff"])
    last_time    = state["last_time"]

    # 若本批第一条与 last_time 不是相邻 10ms，可直接接受（允许缺口），预测时从“最新一条的下一桶”起步
    prev_v = buf[-1] if len(buf) > 0 else None
    for v in vals:
        # 更新差分（需要上一个值）
        if prev_v is not None:
            d = v - prev_v
            diffs.append(d); sdiff += d
            if len(diffs) > D:
                sdiff -= diffs.popleft()
        prev_v = v

        # 更新值窗口
        buf.append(v); ssum += v
        if len(buf) > W:
            ssum -= buf.popleft()

    # 写回状态
    state["sum"] = ssum
    state["sum_diff"] = sdiff
    state["last_time"] = int(x[time_col].iloc[-1])
    if ticker_col in x.columns and x[ticker_col].notna().any():
        state["ticker"] = x[ticker_col].dropna().iloc[-1]

    # —— 预测（向量化） —— #
    return state, _forecast_from_state(state, horizon_ms, time_col, ticker_col)



def _forecast_from_state(state: Dict[str, Any], horizon_ms: int,
                         time_col: str, ticker_col: str) -> pd.DataFrame:
    W, D   = state["W"], state["D"]
    phi    = state["phi"]
    inter  = state["interval_ms"]
    buf    = state["buf"]
    diffs  = state["diffs"]
    last_t = state["last_time"]
    tk     = state["ticker"]

    # 没观测过，无法给时间轴；返回空
    if last_t is None or len(buf) == 0:
        return pd.DataFrame(columns=[time_col, "end_time", "预测总量", "level", "slope", "phi", "Ticker"])

    # level = 最近 W 均值； slope = 最近 D 个差分均值
    level = (state["sum"] / len(buf)) if len(buf) > 0 else 0.0
    slope = (state["sum_diff"] / len(diffs)) if len(diffs) > 0 else 0.0

    # 预测步数（10ms 为单位）
    H = int(horizon_ms // inter)
    k = np.arange(1, H+1, dtype=np.int64)

    # 阻尼累积项 g(k) = (1 - phi^k) / (1 - phi) （phi==1 退化为 k）
    if 0.999999 < phi <= 1.0:
        gk = k.astype(float)
    else:
        gk = (1.0 - np.power(phi, k, dtype=np.float64)) / (1.0 - phi)

    yhat = level + slope * gk
    yhat = np.clip(yhat, 0.0, None)         # 不允许负量（可按需去掉）

    # 未来时间轴
    start_ms0 = _t_to_ms(last_t) + inter     # 预测从“下一桶”开始
    start_ms  = start_ms0 + (k - 1) * inter
    start_t   = _ms_to_t_vec(start_ms)
    end_t     = _ms_to_t_vec(start_ms + inter)

    out = pd.DataFrame({
        time_col: start_t,
        "end_time": end_t,
        "预测总量": yhat.astype(np.float32),
        "level": np.float32(level),
        "slope": np.float32(slope),
        "phi":   np.float32(phi),
        "Ticker": tk
    })
    return out



def _add_ms_hhmmssmmm(t, ms: int) -> int:
    """HHMMSSmmm + ms -> HHMMSSmmm（int）"""
    t = int(float(t))
    h  = t // 10000000
    m  = (t % 10000000) // 100000
    s  = (t % 100000) // 1000
    mm =  t % 1000
    total = (h*3600 + m*60 + s)*1000 + mm + int(ms)
    nh  = (total // 3600000) % 24
    nm  = (total % 3600000) // 60000
    ns  = (total % 60000)  // 1000
    nms =  total % 1000
    return nh*10000000 + nm*100000 + ns*1000 + nms

def slice_first_window(df: pd.DataFrame,
                       duration_s: float,
                       start_col: str = "start_time",
                       end_col: str   = "end_time",
                       by: str = "start"):
    """
    从首行 start_time 起，截取 duration_s 秒的窗口。
    by="start"  -> 取 start_time ∈ [s0, s0+duration) 的行（半开区间）
    by="end"    -> 取 end_time   ≤ s0+duration 的行
    """
    if df is None or df.empty:
        return df

    # 首行起点 & 截止点（+ duration_s 秒）
    s0  = int(float(df.iloc[0][start_col]))
    cut = _add_ms_hhmmssmmm(s0, int(round(duration_s * 1000)))

    # 转成 int64 便于比较（兼容 float 型时间戳）
    s_series = pd.to_numeric(df[start_col], errors="coerce").astype("Int64")
    e_series = pd.to_numeric(df[end_col],   errors="coerce").astype("Int64")

    if by == "end":
        mask = (e_series.notna()) & (e_series.astype(np.int64) <= cut)
    else:  # by == "start"
        s_i64 = s_series.astype(np.int64)
        mask = (s_series.notna()) & (s_i64 >= s0) & (s_i64 < cut)

    return df.loc[mask].copy()


def loop_check_and_print(order_we_sendB5, deal_df, monitor_order,
                         order_id_col="交易所委托号", deal_order_col="叫买序号", time_col="时间"):
    # 预聚合：委托号 -> 是否成交/最早时间
    deal_df = deal_df.copy()
    deal_df[deal_order_col] = deal_df[deal_order_col].astype(str)
    first_time = (deal_df.groupby(deal_order_col, as_index=True)[time_col].min())
    dealt_set = set(first_time.index)

    monitor_order = min(int(monitor_order), len(order_we_sendB5))
    while monitor_order > 0:
        entrust_df = order_we_sendB5.iloc[[monitor_order-1]]
        oid = str(entrust_df[order_id_col].iloc[0])

        if oid in dealt_set:
            print("type error1 认为在n1会被成交，撤单了",
                  f"订单号={oid}, 最早成交时间={int(first_time.loc[oid])}")
            function_log.log_info(fr'{t}_{tk}："type erro1 认为在n1会被成交，撤单了",f"订单号={oid}, 最早成交时间={int(first_time.loc[oid])}"')
                  
        else:
            print("未成交", f"订单号={oid}")
            function_log.log_info(fr'{t}_{tk}：未成交, 订单号={oid}')

        monitor_order -= 1  # 关键：每轮递减直到 0


def get_zt_info(engine, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    从 ticker_high_frequency.limit_stats 表获取涨停数据

    Parameters
    ----------
    engine : SQLAlchemy engine
        数据库连接引擎
    start_date : str
        开始日期，必填，格式如 '20250101' 或 '2025-01-01'
    end_date : str, optional
        结束日期，默认为今天

    Returns
    -------
    DataFrame
        过滤掉 标签='跌停' 的结果
    """
    if not start_date:
        raise ValueError("必须指定 start_date")

    if not end_date:
        # 默认今天
        end_date = datetime.today().strftime("%Y%m%d")

    sql = fr'''
        SELECT *
        FROM ticker_high_frequency.limit_stats
        WHERE DATE >= "{start_date}" AND DATE <= "{end_date}"
    '''

    zt_infor = pd.read_sql(sql, engine)
    each_zt_date = zt_infor[zt_infor['Tag'] != '跌停'].copy()
    each_zt_date.rename(columns={'Date':'日期','Ticker':'代码' ,'Tag':'标签'  },inplace=True )
    each_zt_date['日期'] = each_zt_date['日期'].apply(lambda x: int(str(x).replace('-', '')))
    return each_zt_date

def time_to_int(t):
    h, m = map(int, t.split(":"))
    return h * 10_000_000 + m * 100_000

def get_quantiles_by_ts(df, ts):
    row = df[(df["start_int"] <= ts) & (ts < df["end_int"])]
    
    if row.empty:
        return None
    
    cols = ["10分位数", "20分位数", "30分位数",
            "40分位数", "50分位数", "60分位数",
            "70分位数", "80分位数", "90分位数"]
    
    return row[cols].iloc[0]

# def get_param_by_ts(df, ts):
#     return df.reindex(df.index.union([ts])).sort_index().ffill().loc[ts, 0]

def get_param_by_ts(df, ts):
    s = df.squeeze()   # DataFrame → Series，Series 不变
    s = s.sort_index()
    idx = s.index.values
    

    pos = np.searchsorted(idx, ts, side='right') - 1
    if pos < 0:
        return None
    return s.iloc[pos]

def subtract_ms(current_ms, delta_ms):
    """
    计算当前时间减去指定毫秒数后的时间（当天范围内）。

    参数：
        current_ms (int): 当前时间格式为 HHMMSSmmm 的整数表示（例如 100945000 表示 10:09:45.000）
        delta_ms (int): 要减去的毫秒数

    返回：
        int: 减去 delta_ms 后的时间，以 HHMMSSmmm 格式的整数表示
    """
    # 解析 current_ms 到小时、分钟、秒、毫秒
    milliseconds = current_ms % 1000
    seconds = (current_ms // 1000) % 100
    minutes = (current_ms // 100000) % 100
    hours = (current_ms // 10000000) % 100

    # 计算真正的毫秒数
    true_ms = (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds

    # 减去 delta_ms
    new_true_ms = true_ms - delta_ms
    if new_true_ms < 0:
        new_true_ms = 0  # 不低于当天0点

    # 从新毫秒数换算小时、分钟、秒、毫秒
    new_hours = new_true_ms // (3600 * 1000)
    remainder = new_true_ms % (3600 * 1000)

    new_minutes = remainder // (60 * 1000)
    remainder = remainder % (60 * 1000)

    new_seconds = remainder // 1000
    new_milliseconds = remainder % 1000

    # 组装回 HHMMSSmmm 格式的整数
    return (new_hours * 10000000 +
            new_minutes * 100000 +
            new_seconds * 1000 +
            new_milliseconds)

def _process_one(path,date_each: int, tk: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    单支标的完整流程：
    返回 (stats_df, orders_df)，任一为空则对应位置为 None。
    """
    # tk = '600448.SH'
    # date_each = '20250506'
    # path = fr'Z:\tick\{str(date_each)[:6]}\{date_each}\{tk}'
    
    # 1) 读取并分类
    deal_df, order_df, quote_df = load_tick_data_from_folder(path)
    
    if len(quote_df)==0 or len(deal_df) == 0 or  len(order_df)==0:
        return 
    deal_df, order_df, quote_df = classify_tick_dfs([deal_df, order_df, quote_df])
    
    
    quote_df.columns = colqu
    
    # 2) 930快照（涨停委托）
    entrustzt_price_930 = get_zt_entrust_snapshot_from_path(deal_df, order_df, quote_df)
    if entrustzt_price_930 is None or len(entrustzt_price_930) == 0:
        return   # 无涨停委托

    # 3) 封单时间段 & 去中午休市的持续时间
    aperiod = find_consecutive_zero_ask1_by_rows(quote_df)
    if not aperiod:
        return 

    durations = compute_segment_durations_exclude_mid_break(aperiod)
    if not durations:
        return 

    df_durations = pd.DataFrame(durations, columns=['starttime', 'endtime', 'duration'])
    if df_durations.empty:
        return 

    # 4) 逐段处理（< 14:57:00 的段才参与）
    rows: List[pd.DataFrame] = []
    orders_rows: List[pd.DataFrame] = []

    # 预取委托价（保护：空列/空表）
    zt_price = None
    if '委托价格' in entrustzt_price_930.columns and len(entrustzt_price_930):
        zt_price = entrustzt_price_930['委托价格'].iloc[0]


    for idx, ape in enumerate(aperiod):
        # break
        start_t, end_t = ape
        # break
        # print(ape)
        if start_t >= 145700000:
            continue
        
        # alldf = pd.merge( entrustzt_price_930zt, deal_df,left_on=['交易所委托号'],right_on=['叫买序号'],suffixes=('','_deal') )
        # affdf1 = alldf[alldf['成交价格'] == alldf['委托价格']]
        # af2 = affdf1[affdf1['BS标志'] == 'S']
        # 930 区间截取
        firt_zt = subtract_ms(start_t, 3000)
        after3s_zt =  subtract_ms(start_t, -3000)
        
        # firt_zt = start_t - 10000   # 开始前3秒
        # after3s_zt = start_t + 10000  # 开始后1秒（保持你原逻辑）
        m = (entrustzt_price_930['时间'] >= firt_zt) & (entrustzt_price_930['时间'] <= after3s_zt)
        entrustzt_price_930zt = entrustzt_price_930.loc[m]
        if entrustzt_price_930zt.empty:
            continue

        # 队列跟踪与剩余量重算
        snapshot_df = track_buy_queue_at_limit(entrustzt_price_930zt)
        if snapshot_df is None or snapshot_df.empty:
            continue

        result_df_optimized = recalculate_snapshot_remaining_qty(snapshot_df, deal_df)
        if result_df_optimized is None or result_df_optimized.empty:
            continue

        # 委托编号列表必须非空
        if '委托编号列表' not in result_df_optimized.columns:
            continue
        tmp = result_df_optimized[result_df_optimized['委托编号列表'].apply(_safe_eval_list).map(lambda x: len(x) > 0)]
        if tmp.empty:
            continue

        # 排序 & 取精确封单时刻
        if '时间' not in tmp.columns:
            continue
        tmp = tmp.sort_values('时间').reset_index(drop=True)
        exact_zttime = tmp['时间'].iloc[0]

        # === 统计行 ===
        stat_row = df_durations.iloc[[idx]].copy()
        stat_row['tk'] = tk
        stat_row['buy_date'] = date_each
        stat_row['zt_price'] = zt_price
        stat_row['exact_zttime'] = exact_zttime
        rows.append(stat_row)

        # === 订单行（保护一堆边界）===
        # 以 end_t 作为 "lastzttime"（你原来使用了未定义的 lastzttime）
        lastzttime = end_t

        # 成交过滤区间
        lower_bound = max(exact_zttime, 145600000)
        adf = deal_df[(deal_df['时间'] <= 145659000) & (deal_df['时间'] >= lower_bound)]
        if adf.empty:
            continue

        # 该区间第一笔成交
        first_row = adf.loc[adf['时间'].idxmin()]
        if '叫买序号' not in first_row:
            continue

        ent_order = first_row['叫买序号']
        # 对应委托表过滤
        if '交易所委托号' not in order_df.columns:
            continue
        last_entrust = order_df[order_df['交易所委托号'] == ent_order].copy()
        if last_entrust.empty:
            continue

        # 排除撤单/无效委托
        if '委托类型' in last_entrust.columns:
            last_entrust = last_entrust[last_entrust['委托类型'] != 'D']
        if last_entrust.empty:
            continue

        # 填充统计字段
        last_entrust['start_zttime'] = start_t
        last_entrust['last_zttime'] = end_t
        last_entrust['ztprice'] = entrustzt_price_930['委托价格'].max() if '委托价格' in entrustzt_price_930.columns else None
        last_entrust['after_lastzttime'] = add_milliseconds(lastzttime, 0)
        last_entrust['时间_成交'] = first_row['时间']
        last_entrust['exact_zttime'] = exact_zttime
        last_entrust['tk'] = tk
        last_entrust['buy_date'] = date_each

        orders_rows.append(last_entrust)

    stats_df = pd.concat(rows, ignore_index=True) if rows else None
    orders_df = pd.concat(orders_rows, ignore_index=True) if orders_rows else None
    return stats_df, orders_df, deal_df, order_df, quote_df,entrustzt_price_930

def _safe_eval_list(x):
    """把字符串安全转换为列表；如果本来就是 list 则原样返回；其它返回空表。"""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x) 
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []


def log(msg):
    if VERBOSE:
        print(msg)
        function_log.log_info(msg)
        
class SpeedMeter:
    __slots__ = ("n","bytes","t0","last")
    def __init__(self):
        self.n = 0
        self.bytes = 0
        self.t0 = time.perf_counter()
        self.last = self.t0

    def add(self, size:int)->None:
        self.n += 1
        self.bytes += size

    def report(self, tag:str=""):
        now = time.perf_counter()
        dt_total = now - self.t0
        dt_seg   = now - self.last
        rps      = self.n / dt_total if dt_total>0 else 0.0
        mbps     = (self.bytes/dt_total)/(1024*1024) if dt_total>0 else 0.0
        avg_us   = (dt_total/max(self.n,1))*1e6
        print(f"[{tag}] ticks={self.n}, time={dt_total:.3f}s, avg={avg_us:.1f}µs/tick, "
              f"throughput={rps:.1f} ticks/s, tx={mbps:.2f} MiB/s, "
              f"avg_size={self.bytes/max(self.n,1):.1f} B")
        self.last = now
        return {"dt_total": dt_total, "dt_seg": dt_seg, "rps": rps, "mbps": mbps, "avg_us": avg_us}

        
colqu = ['万得代码', '交易所代码', '自然日', '时间', '成交价', '成交量', '成交额', '成交笔数', 'IOPV',
       '成交标志', 'BS标志', '当日累计成交量', '当日成交额', '最高价', '最低价', '开盘价', '前收盘', '申卖价1',
       '申卖价2', '申卖价3', '申卖价4', '申卖价5', '申卖价6', '申卖价7', '申卖价8', '申卖价9', '申卖价10',
       '申卖量1', '申卖量2', '申卖量3', '申卖量4', '申卖量5', '申卖量6', '申卖量7', '申卖量8', '申卖量9',
       '申卖量10', '申买价1', '申买价2', '申买价3', '申买价4', '申买价5', '申买价6', '申买价7', '申买价8',
       '申买价9', '申买价10', '申买量1', '申买量2', '申买量3', '申买量4', '申买量5', '申买量6', '申买量7',
       '申买量8', '申买量9', '申买量10', '加权平均叫卖价', '加权平均叫买价', '叫卖总量', '叫买总量', '不加权指数',
       '品种总数', '上涨品种数', '下跌品种数', '持平品种数', 'Unnamed: 66']

#%% ade = deal_df[deal_df['成交代码']=='C']
# 17min/10只股票
from sqlalchemy import create_engine

save_dir = fr'C:\Users\fyx90\Desktop\excel\打板回测\queue'
function_log = CustomLogger(save_dir)

# ztinfor_path = r'C:\Users\fyx90\Desktop\excel\涨跌停炸板数据汇总3.xlsx'
# zt_infor = pd.read_excel(ztinfor_path)

# startbt =20250501
# endbt = 20250601
# zt_infor1 = zt_infor[zt_infor['日期'] >=startbt]
# each_zt_date = zt_infor1[zt_infor1['日期'] <=endbt]

engine = create_engine('mysql+pymysql://ainvest_all_read:ainvest_all_read1@rm-2zewagytttzk6f24xno.mysql.rds.aliyuncs.com:3306/')

import glob
import os
import re

# dir_path = r"F:\parameter"
dir_path = r"C:\Users\fyx90\Desktop\excel\打板回测\bt_data\result"
namepara = 'loose1'
xlsx_files = [
    f for f in os.listdir(dir_path)
    if f.endswith(fr"{namepara}.xlsx")
]

# date_pattern = re.compile(r'(\d{8})_(\d{8})')
# date_pattern = re.compile(
#    fr'order_deal_result_(\d{8})_(\d{8})_({namepara})'
# )
pattern = re.compile(
    fr'order_deal_result_(\d{{8}})_(\d{{8}})_{namepara}\.xlsx'
)

results = []

# for fname in os.listdir(xlsx_files):
#     if not fname.endswith('.xlsx'):
#         continue

#     m = re.search(r'(\d{8})_(\d{8})_({namepara})', fname)
#     if m:
#         start_date, end_date, _ = m.groups()
#         results.append((fname, start_date, end_date))
        
for fname in os.listdir(dir_path):
    # 排除临时文件
    if fname.startswith('~$'):
        continue

    m = pattern.match(fname)
    if m:
        start_date, end_date = m.groups()
        results.append({
            'file': fname,
            'start_date': start_date,
            'end_date': end_date
        })



df_res = pd.DataFrame(results)
df_res['next_bt'] = df_res['end_date'].shift(-1)

entrust_queue_before = pd.DataFrame()
entrust_queue_after = pd.DataFrame()

for idx,row in df_res[0:1].iterrows():
    
    print(row)
    startbt = row['start_date']
    endbt =  row['end_date']
    nextbt = row['next_bt']
    
    endbt = '20240101'
    nextbt= '20240201'
    each_zt_date = get_zt_info(engine,endbt,nextbt)


    # each_zt_date = zt_infor[zt_infor['日期'] ==20250402]
    each_zt_date = each_zt_date[each_zt_date['标签']!='跌停']
    total_trades=pd.DataFrame()
    
    df_ts = pd.DataFrame()
    date1 = each_zt_date['日期'].drop_duplicates()
    total_benchmark_trades = pd.DataFrame()
    df_demo_test = pd.DataFrame()
    
    
    # 先把 t 列单独取出来 
    # order_deal_result_20240401_20240501_loose1
    # para =pd.read_excel(dir_path + fr'\order_deal_result_20240401_20240501_loose1.xlsx')
    para =pd.read_excel(dir_path + fr'\order_deal_result_{startbt}_{endbt}_loose1.xlsx')
    
    para_ordertime =pd.read_excel(dir_path + fr'\order_deal_result_{startbt}_{endbt}_loose1.xlsx',sheet_name='order_time_all')
    
    para_ordertime[["start", "end"]] = para_ordertime["时间段"].str.split("-", expand=True)
    para_ordertime["start_int"] = para_ordertime["start"].apply(time_to_int)
    para_ordertime["end_int"] = para_ordertime["end"].apply(time_to_int)
    
    t_values = para["t"]
    
    # 对每列（除 t）应用
    result = {
        col: para.loc[para.index[para[col] < 0][-1], "t"] if (para[col] < 0).any() else None
        for col in para.columns if col != "t"
    }
    result1 =pd.DataFrame(result,index=['0']).T
    result1 = result1.fillna(0)
    
    
    # 对每列（除 t）应用
    result2 = {
        col: para.loc[para.index[para[col] < 0.2*0.01][-1], "t"] if (para[col] <  0.2*0.01).any() else None
        for col in para.columns if col != "t"
    }
    result2 =pd.DataFrame(result2,index=['0']).T
    result2 = result2.fillna(0)

save_dir = fr'C:\Users\fyx90\Desktop\cpp\bt_log\bt_simulate_{startbt}_{endbt}'

function_log = CustomLogger(save_dir)

#%%
import gc

# min1 =60
# min5 =60*5
# min15 = 60*15
timedeal_total = pd.DataFrame()
timedeal_list = []
# ===== 全局变量：存储所有时间戳的pred_next和final_df记录 =====
# pred_next_history = pd.DataFrame()
# final_df_history = pd.DataFrame()

times =time.time()
pred_next = 100

VERBOSE = True  # 或者做成函数参数
benchn = 1

# 叠加热门股股票池
# stookpool = pd.read_excel(fr'E:\temp\bt_res\stock_pool2025.xlsx')
# stookpool['buy_Date'] = stookpool['As_Of_Date'].apply(lambda x : int(str(x).replace('-','')[0:8]))

engine = create_engine(
    'mysql+pymysql://cn_ainvest_db:cn_ainvest_sd3a1@rm-2zewagytttzk6f24xno.mysql.rds.aliyuncs.com:3306'
)    

date_end_date = pd.to_datetime(endbt, format='%Y%m%d')
date_nextbt = pd.to_datetime(nextbt, format='%Y%m%d')

sqlyz = fr'''
        SELECT *
        FROM ticker_high_frequency.stock_pool where as_of_date >="{date_end_date}" and as_of_date <="{date_nextbt}"
    '''
            

stookpool = pd.read_sql(sqlyz, engine)
stookpool['buy_Date'] = stookpool['As_Of_Date'].apply(lambda x : int(str(x).replace('-','')[0:8]))

# 是否包括一字
sql_yz = fr'''
        SELECT *
        FROM ticker_high_frequency.ban_stock_condition where as_of_date >="{date_end_date}" and as_of_date <="{date_nextbt}"
    '''
banstock = pd.read_sql(sql_yz, engine)
# stookpool['buy_Date'] = stookpool['As_Of_Date'].apply(lambda x : int(str(x).replace('-','')[0:8]))
          
# lbticker=  pd.read_excel(fr'C:\Users\fyx90\Desktop\cpp\maxban_ticker.xlsx')

sql_lbt = fr'''
        SELECT *
        FROM ticker_high_frequency.ban_stock_maxnumofzt
    '''
lbticker=  pd.read_sql(sql_lbt, con=engine)

dates = lbticker['Date'].drop_duplicates()

# 最大持股数量10只
max_nstock = 10

banstock['As_Of_Date'] = pd.to_datetime(banstock['As_Of_Date'])
#%%
# n1危险期 = 1 要不然n=0炸板了
# 判断总队列是否会被炸
spd = SpeedMeter()

strong = 0
log_buf = [] 
lb_window= 300
#  20240325 20240326/ 20240327 20240328 20240329
for date_each in date1[15:16]:
    nstock = 0
    print(date_each)
    # date_each  = 202403s01
    # break
    each_zt_date1 = each_zt_date[each_zt_date['日期']==date_each]
    stock_pool_today = stookpool[stookpool['buy_Date']== date_each]
    stock_pool_today_list = stock_pool_today['Ticker'].tolist()
    
    each_zt_date1 = each_zt_date1.sort_values(by=['First_Limit_Time'])
    # each_zt_date1 = merged_data[merged_data['buy_date']==date_each]
    # date_each = 20260114
    date_each1 = pd.to_datetime(date_each, format='%Y%m%d')
    # function_log.log_info(fr'trade_date:{date_each}')
    #%
    #% countt =0 
    for tk in each_zt_date1['代码'].tolist()[0:]:
        # print(tk)
    # for tk in ts_list:
            # gc.collect()
    # for tk in tsample:
            # tk = '000980.SZ'
            banornot = banstock[(banstock['Ticker'] == tk) & (banstock['As_Of_Date'] == date_each1) ]
            
            date_eachstart = date_each1 - timedelta(3)
            date_eachend = dates[dates<date_each1].iloc[-2]
            
            list_stock = lbticker[(lbticker['Date']<date_each1) & (lbticker['Date']>=date_eachend) ]
            list_stock1 = list_stock[list_stock['ban_ticker']==True]
            
            if tk in list_stock1['Ticker'].tolist():
                print(fr'{tk} 连板危险性较大，不做')
                continue
            
            # banornot = banstock[banstock['As_Of_Date'] == pd.to_datetime(date_each1) ]
        
            if len(banornot):
                continue
            
            if nstock >= max_nstock:
                break
        # try:
            
            # tk = "000509.SZ"
            print(tk)
            if tk in stock_pool_today_list:                
                # continue
                stock_pool_today_tk = stock_pool_today[stock_pool_today['Ticker']==tk]
                prob = stock_pool_today_tk['涨停累计']/(stock_pool_today_tk['炸板累计']+stock_pool_today_tk['涨停累计'])
                prob1 = float(prob.iloc[0])
            else:
                prob1=0
                
            strong = 0
             
             # 判断是哪个交易所
            if 'SH' in tk:
                sending_interval = 200
                sending_interval = 10
            elif 'SZ' in tk:
                sending_interval = 10
            # sending_interval = 0    
                
            if prob1 >0.7:
                print(fr'{tk}为强势股')
                strong = 1
                n=0
                # continue
                
            else:
                strong = 0
                n=5
                # continue
                
            # tk = '002004.SZ'
            label = each_zt_date1[each_zt_date1['代码']==tk]['标签'].iloc[0]
            # function_log.log_info(fr'{tk}_{each_zt_date1}')
            #%         
            # path = r'F:\tick\202504\20250408\601008.SH'
            # date_each = '20240401'
            path = fr'F:\tick\{str(date_each)[0:6]}\{date_each}\{tk}'
            # path = fr'Y:\{str(date_each)[0:6]}\{date_each}\{tk}'
            # path = fr'Y:\{str(date_each)[0:6]}\{date_each}\{tk}'
            # Y:\202601
            # ab = deal_df[deal_df['叫买序号']==45597995]
            #deal_df[deal_df['叫卖序号']==46942262] 45597995

                # 多加一个延迟
                # sending_interval = 110
                
            function_log.log_info(f'下一只股票：{tk} 的数据推送时间为{sending_interval}ms')
            
            # 逐笔快照合成，判断涨停时间 
          
            resultdata = _process_one(path,date_each, tk)
            
            if resultdata is None:
                continue   # 或者 break / pass / logging
            else:
                stats_df, orders_df, deal_df, order_df, quote_df, entrustzt_price_930 = resultdata
            
            ztprice = entrustzt_price_930['委托价格'].max() # 要计算得出
            # cancel_order = deal_df[deal_df['成交代码']=='C']['叫买序号'].tolist()

            deal_df = deal_df[deal_df['自然日']!=0]
          
            # entrustzt_price_930_v1 = entrustzt_price_930[entrustzt_price_930['交易所委托号'].apply(lambda x : x not in cancel_order)]
            
            if len(entrustzt_price_930)==0:
                print(fr'no zt {tk}')
                function_log.log_info(fr'no zt {tk}')
                continue
            
            if stats_df is None or stats_df['starttime'].iloc[0] < 93103000:
                # print(len(stats_df))
                print(fr'{tk}一字开 不排板！')
                function_log.log_info(fr'{tk}一字开 不排板！')
                continue
                
           
            if len(stats_df)==0:
                continue
            
            # countt = countt+1 
            
            #%
            done = False
            for i in range(0,len(stats_df)):
                print(i)
                if done:
                    break
                # print(i)
                start_sending = stats_df.iloc[i]['exact_zttime']
                # stime=stats_df.iloc[i]['starttime']
                etime=stats_df.iloc[i]['endtime']
                # ts_list = gen_timestamps(start_sending)
                # time1 = time.time()
                # ztprice = entrustzt_price_930['委托价格'].max() # 要计算得出
                # entrustzt_price_queue = []
                #% 交易所延迟+本身的延迟
                send_order = extract_first_duration_and_add1(start_sending,seconds=(sending_interval+0)/1000) # 加了延迟
                state2 = init_limit_buy_state(ztprice)
                state3 = init_limit_buy_state(ztprice)
                # print(state2)
                # active_snapshots = pd.DataFrame()
                # TIMINGS = []
                # function_log.log_info(f'{tk}:涨停时间：{start_sending}')
                
                # 模拟挂单
                order_we_send = entrustzt_price_930[entrustzt_price_930["时间"] >= send_order]
                
                # 找到不撤单的单子
                order_we_sendB =order_we_send[order_we_send['委托代码']=='B']          
                #order_we_sendB = order_we_sendB[order_we_sendB['委托类型']!='D']
                # order_we_sendB1 = order_we_sendB.drop_duplicates(subset=['委托数量','交易所委托号'],keep=False)            
                order_we_sendB1 = order_we_sendB.drop_duplicates(subset=['时间'],keep='first')
                
                order_we_sendB1 = order_we_sendB1[order_we_sendB1['委托类型']!='D']
                
                #% 连续挂出5个排涨停的单子
                start_time = order_we_sendB1.iloc[0]['时间']   
                          
                result_ot = get_quantiles_by_ts(para_ordertime, start_time)
                         
                order_we_sendB5 = pd.DataFrame()
                time_list = []
                
                for i in result_ot[2:7]:
                     # break
                    # print(i)
                    if strong:
                        i = 0
                    else:
                        i = i
                        
                    ordertime = int(add_milliseconds(start_time, i))
                    ordertime_end = int(add_milliseconds(start_time, i+10)) # 100毫秒的冗余
                    # ordertime_start = int(add_milliseconds(start_time, i-20))
                    # 一次性过滤：时间 >= ordertime 且 不在 time_list
                    candidate = order_we_sendB1[
                        (order_we_sendB1['时间'] >= ordertime) &
                        (order_we_sendB1['时间'] <= ordertime_end) &
                        (~order_we_sendB1['时间'].isin(time_list))
                    ]
                
                    if candidate.empty:
                        continue   # 或 break，看你的业务逻辑
                
                    order_we_sendB5_each = candidate.iloc[[0]]
                
                    time_list.append(order_we_sendB5_each['时间'].iloc[0])
                    order_we_sendB5 = pd.concat([order_we_sendB5, order_we_sendB5_each])
        
                function_log.log_info(f'{tk}:下的五单子时间信息，{order_we_sendB5}')
                if len(order_we_sendB5)==0:
                    continue
                
                # ts_list1 = entrustzt_price_930
                snap_inc_total=snap_inc =pd.DataFrame()
                state1 = None
                pad_state =roll_state = roll_state1= None
                fc_state = inc_padded = inc = None
                pred_next = consum = None
          
                # n1 n2时间获取
                n1 = get_param_by_ts(result1, start_time)+1
                n2 = get_param_by_ts(result2, start_time)+1
                ordlist= order_we_sendB5['交易所委托号'].tolist()
                # rat = 0
                if benchn:
                    # 非强势股参数
                    n1 = n
                    n2 = 10
                    
                # if strong:
                #     n1 = n2 = 0
                #     # order_we_sendB5['']
                #     # timedeal = deal_df_by_order_id.loc[[ordlist]].reset_index()
                #     # timedeal = deal_df[deal_df.index.isin(ordlist)].reset_index()
                #     #for numen in ordlist:
                #     timedeal = deal_df[deal_df['叫买序号'].isin(ordlist)].reset_index()
                #     if len(timedeal):
                #         timedeal['标签'] = label
                #         timedeal_list.append(timedeal)
                #         nstock = nstock+1
                #     continue
                
                print('start queuing')
                n1_time = add_minutes_to_custom_time(start_time, n1)
                n2_time = add_minutes_to_custom_time(start_time, n2)
                
                total_queue = total_pred_1 = pd.DataFrame()
                
                monitor_order = 0
                board2 = board3 = 0
                monitor_order_num_total= len(order_we_sendB5)
                
                # ===== 优化：预创建ordlist_set，避免每次循环重新创建 =====
                ordlist_set = set(ordlist)
                
                #%
                # ts_listsub = [x for x in ts_list if x < 93051070]
                
                quick_judge = int(add_milliseconds(start_sending, 100))
                count = start_queue = 0
                
                # ===== 性能优化：预索引DataFrame，避免每次循环O(n)过滤 =====
                # 按时间分组，创建字典索引
                # order_df_grouped = {t: g for t, g in order_df.groupby('时间')}
                
              
                entrustzt_price_930_exc = entrustzt_price_930[~(
                        (entrustzt_price_930['委托类型'] == 'D') &
                        (entrustzt_price_930['交易所委托号'].isin(ordlist))
                    )]
                
                deal_df_exc = deal_df[~(
                        (deal_df['成交代码'] == 'C') &
                        (deal_df['叫买序号'].isin(ordlist))
                    )]
                deal_df_exc = deal_df_exc[(deal_df_exc['时间']>=start_sending) & (deal_df_exc['时间']<=etime)]
                entrustzt_price_930_exc = entrustzt_price_930_exc[(entrustzt_price_930_exc['时间']>=start_sending) & (entrustzt_price_930_exc['时间']<=etime)]
                
                order_df_grouped = {t: g for t, g in entrustzt_price_930_exc.groupby('时间')}
                deal_df_grouped = {t: g for t, g in deal_df_exc.groupby('时间')}
                
                # 获取所有事件时间并排序
                event_times = np.unique(
                        np.concatenate([
                            order_df['时间'].to_numpy(np.int64, copy=False),
                            deal_df['时间'].to_numpy(np.int64, copy=False),
                        ])
                    )
                
                event_times1 = event_times[event_times >= start_sending]
                
                # ===== 优化：预索引deal_df的叫买序号列，加速查找 =====
                deal_df_by_order_id = deal_df.set_index('叫买序号') if '叫买序号' in deal_df.columns else None
                
                # ===== 优化：用列表累积timedeal，最后一次性concat =====
                # ===== 记录每个时间戳的pred_next和final_df =====
                timestamp_records = []  # 存储每个时间戳的记录
                
                #%
                zb_done = False
                for t in event_times1:
                    # break
                    if zb_done:
                        break
                    if len(ordlist) == 0:
                        break  # 全部单子已经处理完，直接结束
                    
                    # 使用预索引字典，O(1)查找
                    order_df_interval = order_df_grouped.get(t, pd.DataFrame())
                    deal_df_interval  = deal_df_grouped.get(t, pd.DataFrame())
                    
                    if order_df_interval.empty and deal_df_interval.empty:
                        continue
                    # log_time("slice_order", t, (time.perf_counter() - t0) * 1000, {"rows": len(order_df_interval)})
                    # 小模型，防止炸板 计算前50毫秒的队列，要不要挂单
                    
                    
    
                    
                    if not order_df_interval.empty or not deal_df_interval.empty:
                        
                        snap_inc = step_limit_buy_with_deals(
                                state=state2,
                                delta_orders=order_df_interval,
                                delta_deals=deal_df_interval,
                                # 如果你的委托号是整型列，设 True 性能更好：
                                order_id_is_int=True,
                                # U 的含义：'override' 覆盖为新量；'delta' 代表在原量基础上追加
                                update_mode='override',
                                emit_empty=False,        # 队列为空时是否也产出一行快照
                                sort_ids=False           # 输出队列顺序：False 保持 FIFO；True 按编号升序
                            )
                        
                        if len(snap_inc):
                            totalqueue = snap_inc['总委托数量'].iloc[0]
                            
                            id_queue = snap_inc['委托编号列表'].iloc[0]
                            ordsend = order_we_sendB5['交易所委托号'].tolist()
                        
                        # set(id_queue).issubset(set(ordsend))

    
                            if set(id_queue).issubset(set(ordsend)):
                                entrust_df = order_we_sendB5.iloc[0]      
                                entrust_df['成交价格'] = ztprice
                                timedeal_list.append(entrust_df)
                                print(fr'炸板没撤单')
                                nstock = nstock+1
                                done=True
                                break
                                                    
                        # snap_inc_total = pd.concat([snap_inc_total,snap_inc])
                        # print(fr"t:{t} , snap:{snap_inc['每笔数量列表'].map(sum).sum()}")
                        # print(snap_inc['总委托数量'])
                        # 小模型，防止炸板 计算前50毫秒的队列，要不要挂单
                        if count==0:
                            start_queue = snap_inc['每笔数量列表'].map(sum).sum()                   
                            count = 1
                            print(start_queue)
                            log(fr'{t}, get start queue: {start_queue}')
                            
                        if t >= quick_judge and count==1:
                            diff = start_queue - snap_inc['每笔数量列表'].map(sum).sum()  
                            print(diff)
                            log(fr'{t}, get end queue: {snap_inc["每笔数量列表"].map(sum).sum()}')
                            count = 2
                            # break
                            if diff >0:
                                log(fr'{t}_{tk}：50毫秒队列下降，先不挂单')
                                print(fr'{t}_{tk}：50毫秒队列下降，先不挂单')
                                break
                            
                        
                        # 每个tick判断是不是被成交了
                        if len(deal_df_interval):
                            deal_df_interval1 = deal_df_interval[deal_df_interval['成交代码'] != 'C']
    
                            if deal_df_interval1 is not None and ordlist:
                                hit_ids = deal_df_interval1.index.intersection(ordlist)
                                # 如何是要撤的单子，排在前面的怎么办
                                if len(hit_ids) > 0:
                                    oid = hit_ids[0]   # 第一个成交的订单号
                            
                                    final1 = deal_df_interval1.loc[[oid]].reset_index()
                            
                                    timedeal_list.append(final1)
                                    nstock += 1
                            
                                    log(rf'第{monitor_order+1}单成交')
                                    print(final)
                            
                                    done = True
                                    break
   
                        #% print(snap_inc) 
                        # snap_inc_total = pd.concat([snap_inc_total,snap_inc])
                    
                        
                        # 每10ms，我们单子前面有多少挂单
                        # entrust_df = order_we_sendB5[order_we_sendB5['交易所委托号']==idx]
                        # 0是第一笔挂单
                        # entrust_df = order_we_sendB5.iloc[[monitor_order]]
                        # idx = entrust_df['交易所委托号'].iloc[0]
                        # # zt后有多少单子排在前面
                        # result_df = compute_waiting_qty_before_entrust(snap_inc, entrust_df)
                        
                        # if t==result_df
                        # result_df = pd.concat(
                        #                 [
                        #                     compute_waiting_qty_before_entrust(
                        #                         snap_inc,
                        #                         entrust_df := order_we_sendB5.iloc[[i]]
                        #                     ).assign(
                        #                         monitor_order=i,
                        #                         entrust_id=entrust_df['交易所委托号'].iloc[0]
                        #                     )
                        #                     for i in range(monitor_order, len(order_we_sendB5))
                        #                 ],
                        #                 ignore_index=True
                        #             )
    
                        # 优化版本：使用set提高查找效率，使用列表推导式保持代码简洁
                        # 注意：ordlist_set已在循环外预创建，这里只需更新（当ordlist变化时）
                        # print('order')
                        if len(ordlist_set) != len(ordlist):
                            ordlist_set = set(ordlist)
                        
                        # 先创建所有需要处理的索引和对应的委托号
                        to_process = [
                            (i, order_we_sendB5.iloc[i]['交易所委托号'])
                            for i in range(monitor_order, len(order_we_sendB5))
                            if order_we_sendB5.iloc[i]['交易所委托号'] in ordlist_set
                        ]
                        
                        # 只处理符合条件的委托
                        result_df = pd.concat(
                            [
                                compute_waiting_qty_before_entrust(
                                    snap_inc,
                                    order_we_sendB5.iloc[[i]]
                                ).assign(
                                    monitor_order=i,
                                    entrust_id=entrust_id
                                )
                                for i, entrust_id in to_process
                            ],
                            ignore_index=True
                        )
                                    
                
                        # result_df = result_df.dropna(subset=['排队数量'])
                        
                        #%
                        # res_total  = pd.concat([res_total,result_df])
                        
                        # 还没下单，所以先计算第一单的排队数量，就是现在的总队列的数量, 如何实际消耗量大于预测消耗量，则不下单！
                        # 需要改进？在下次封板后计算下单？？？？
                        if result_df.empty or (not result_df.empty and result_df['排队数量'].iloc[0] is None):
                            if result_df.empty:
                                # 如果result_df为空，创建一个默认行
                                result_df = pd.DataFrame([{'排队数量': snap_inc['每笔数量列表'].map(sum).sum(), '时间': t}])
                            else:
                                result_df['排队数量'] = snap_inc['每笔数量列表'].map(sum).sum()
                        
                        # else:
                        #     continue
                        # 每10ms，消耗的成交的单子是多少
                        lst = ordlist[0]
                        max_id = lst if lst else 1
                    
                        state1, inc = analyze_deal_cancel_lazy(
                                    state1, 
                                    idx=max_id, 
                                first_over_60= start_sending, tk=tk, interval_ms=10,
                                deal_df=deal_df_interval,
                                entrustzt_price_930=order_df_interval,  # 这里传本步的"涨停挂单/撤单"增量
                                # adjust_outliers=adjust_outliers, window=20, nstd=3,  # 需要平滑时打开
                                emit_full=False   # 想看累计全量就设 True
                            )
                        
                        pad_state, inc_padded = pad_10ms_lazy(
                                pad_state,
                                inc_df=inc,
                                interval_ms=10,
                                base_start_time=start_sending  # 首次建议给个起点，之后可不传
                            )
                        # 最终全量结果
                        # final_df  = state1["result"]
                        final_df = pad_state["df"].reset_index()[["start_time","end_time","成交总量_撤单","成交总量","总量","Ticker","总量调整"]]
                            
                        # final_df['总量调整'].sum()
                        # 反推下一个10ms的成交
                        # pred为下10ms的成交
                        # 有一个模型预测未来100ms会不会被成交
                        # if len(result_df)==0:
                        #     continue order_we_sendB5.iloc[[i]]
                        
                        # 优化：检查result_df是否为空
                        if result_df.empty:
                            continue
                        t_order = result_df.iloc[0]['时间']
                       
                        
                        if not inc.empty :
                            # 过去100个10毫秒的平均,有大单冲击误撤
                            # consum = inc['总量调整'].iloc[0]     
                            final_df_Set= final_df[final_df['start_time']>quick_judge]
                            consum = final_df_Set['总量调整'].iloc[-lb_window:].mean()
                            consum_last = final_df_Set['总量调整'].iloc[-1:].sum()
                            
                            
                            if consum_last/totalqueue>0.5 and t>quick_judge:
                                print('队列可能要炸板，全撤单')
                                print(consum_last/totalqueue)
                                zb_done= True
                                # 是不是炸了被成交了？
                                
                                
                            # breakfinal_df
                            # print(consum)
                            #% 判断n1时间之内是否会成交
                            # 注意：pred_next 需要先计算才能使用
                            # 如果pred_next还没有计算（第一次循环），先跳过判断
                            if pred_next is None or (isinstance(pred_next, pd.Series) and len(pred_next) == 0):
                                # pred_next还未计算，继续到后面计算
                                pass
                            elif isinstance(pred_next, pd.Series) and len(pred_next) > 0:
                                if t < n1_time and t >= quick_judge: # 炸板 
                                    # 如果没来的及检测怎么办？
                                    for indx, prd in pred_next.items():
                                        # print(fr"indx:{indx}")
                                        # print(fr"prd:{prd}")
                                        # print('compare')
                                        if consum>prd and indx in ordlist: # and consum< pred_last:
                                            # if total_pred >= queue:
                                            # print(fr'{consum}, {pred_next}')
                                            # print(fr'{t}：n1时间内要成交，应该撤第{monitor_order+1}单！')
                                            # function_log.log_info(fr'{t}_{tk}：n1时间内要成交，应该撤第{monitor_order+1}单！')
                                            
                                            log(fr'{t}：n1时间内要成交，应该撤第{monitor_order+1}单！')
                                            # print(consum)
                                            # print(prd)
                                            
                                            # done= True
                                            # ent_id = order_we_sendB5.iloc[monitor_order]['交易所委托号']
                                            monitor_order += 1
                                            # if indx in ordlist:
                                            ordlist.remove(indx)
                                         # 是不是加一个判断，被成交的判断，但如果没撤，后面也会成交
                                            # done=True
        
                                        if monitor_order == monitor_order_num_total:
                                            log(fr'{t}：n1时间{len(order_we_sendB5)}单全撤了')
                                            # print(fr'{t}：n1时间{len(order_we_sendB5)}单全撤了')
                                            # function_log.log_info(fr'{t}_{tk}：n1时间{len(order_we_sendB5)}单全撤了')
                                            loop_check_and_print(order_we_sendB5, deal_df, monitor_order)
                                            
                                            break
                                            
                                            # break
                                    # elif consum > pred_last:
                                    #     print(fr'{t}：n1时间内要成交，应该要全部撤单！')
                                    #     function_log.log_info(fr'{t}_{tk}：n1时间内要成交，应该要全部撤单！')
                                    #     break
                                    
                                    else:
                                        # print('n1时间不会成交，不用撤单')
                                        pass
                                
                                if t>= n1_time and t < n2_time and len(ordlist):
                                # print('1')    
                                    # for indx, prd in pred_next.items():
                                        # print(pred_next)
                                        # print(indx)
                                        pred1 = pred_next.iloc[0]
                                        if consum>pred1 : # 有可能被成交
                                            # print('prob2')
                                            if len(ordlist)==1: # 最后一单
                                                if board2 ==0:
                                                    log(fr'{t}：n1和n2之间，剩下最后一单，不再进行撤单1')
                                                    board2 =1 
                                                # print(fr'{t}：n1和n2之间，剩下最后一单，不再进行撤单1')
                                                # entrust_id = entrust_df['交易所委托号'].iloc[0]
                                                # 优化：使用预索引加速查找
                                                deal_real = deal_df_by_order_id[deal_df_by_order_id['成交代码']!='C']
                                                if deal_df_by_order_id is not None and ordlist[0] in deal_real.index:
                                                    final = deal_real.loc[[ordlist[0]]].reset_index()
                                                    if not final.empty:
                                                        timedeal_list.append(final)
                                                        nstock = nstock+1
                                                    
                                                    # if final.empty:
                                                    #     will_deal = consum < pred1
                                                    #     log(f"{t}_{tk}：n2时间内这单{monitor_order+1}"
                                                    #         f"{'会' if will_deal else '不会'}成交,最终没有成交")
                                                    #     # function_log.log_info(
                                                    #     #     f"{t}_{tk}：n2时间内这单{order_no}"
                                                    #     #     f"{'会' if will_deal else '不会'}成交,最终没有成交"
                                                    #     # )
                                                    #     break
                                                    
                                                    
                                                        log(rf'第{monitor_order+1}单成交')
                                                        print(final)
                                                        done = True
                                                        break
                                                
                                                else:
                                                    
                                                    # 是否是最后时刻封死涨停
                                                    if etime < 145600000:
                                                        continue
                                                            
                                                    # 算一下队列是否被消耗掉
                                                    snap_inc_total = step_limit_buy_with_deals(
                                                            state=state3,
                                                            delta_orders=entrustzt_price_930_exc,
                                                            delta_deals=deal_df_exc,
                                                            # 如果你的委托号是整型列，设 True 性能更好：
                                                            order_id_is_int=True,
                                                            # U 的含义：'override' 覆盖为新量；'delta' 代表在原量基础上追加
                                                            update_mode='override',
                                                            emit_empty=False,        # 队列为空时是否也产出一行快照
                                                            sort_ids=False           # 输出队列顺序：False 保持 FIFO；True 按编号升序
                                                                )
                        
                                                    # fakeid = entrust_df['交易所委托号'].iloc[0]  
                                                    fakeid =result_df['交易所委托号'].iloc[0]
                                                    # minid = snap_inc_total['委托编号列表'].iloc[0][0]
                                                    lst = snap_inc_total['委托编号列表'].iloc[0]

                                                    minid = next((x for x in lst if x != fakeid), None)

                                                    
                                                    
                                                    id_queue = snap_inc_total['委托编号列表'].iloc[0]
                                                    ordsend = order_we_sendB5['交易所委托号'].tolist()
                                                    
                                                    # set(id_queue).issubset(set(ordsend))
                                                    

                                                    if set(id_queue).issubset(set(ordsend)):
                                                        continue
                                                    # result_df2 = compute_waiting_qty_before_entrust(snap_inc, entrust_df)
                                                    # queue2 = result_df2['排队数量'].iloc[0]
                                                    # if queue2>0:
                                                    #     pass
                                                    
                                                    if fakeid < minid :
                                                        print('最终成交')
                                                        log(rf'第{monitor_order+1}单成交,计算消耗队列')
                                                        entrust_df['成交价格'] = ztprice
                                                        timedeal_list.append(entrust_df)
                                                        nstock = nstock+1
                                                        done=True
                                                        break
                                                    else:
                                                        print('最终没成交')
                                                        done=True
                                                        break
                                                        
                                                    # final = pd.DataFrame()
                                                
                                            
                                            # if len(final):
                                            #     break
                                            
                                            
                                            if  len(ordlist)>1: # and consum< pred_last:
                                                # 查看下一单的委托编号
                                                entrust_df = order_we_sendB5.iloc[[0]]
                                                # 计算下一单的队列
                                                result_df1 = compute_waiting_qty_before_entrust(snap_inc, entrust_df)
                                                queue = result_df1['排队数量'].iloc[0]
                                                
                                                
                                                if queue is None:
                                                    # print('被炸了')
                                                    numen = entrust_df['交易所委托号'].iloc[0]
                                                    # 优化：使用预索引加速查找s
                                                    deal_real = deal_df_by_order_id[deal_df_by_order_id['成交代码']!='C']
                                                    if deal_df_by_order_id is not None and numen in deal_real.index:
                                                        
                                                        timedeal = deal_real.loc[[numen]].reset_index()
                                                        timedeal['标签'] = label
                                                        timedeal_list.append(timedeal)
                                                        nstock = nstock+1
                                                        done=True
                                                        break
                                                    else:
                                                        # log(f"{t}: 委托 {numen} 炸板但未成交")
                                                        print('caculate queue')
                                                    # break
                                                
                                                if queue is not None:   
                                                
                                                        # 预测需要多少个10ms成交
                                                    # pred2 = pred_next.iloc[0]
                                                    consumsell = final_df_Set['成交总量'].iloc[-100:].sum() 
                                                    if consumsell==0:
                                                        print('无订单消耗')
                                                        continue 
                                                    else:
                                                        deal_time =_add_ms_hhmmssmmm(t, queue/consumsell*1000 )
                                                
                                                    if deal_time > 145700000 : 
                                                        log(fr'{t}：{monitor_order+1}单在145700000之前预测无法成交，不撤单')
                                                        
                                                        # print(fr'{t}：{monitor_order+1}单在145700000之前预测无法成交，不撤单')
                                                        # function_log.log_info(fr'{t}_{tk}：{monitor_order+1}单在145700000之前预测无法成交，不撤单')
                                                    elif deal_time < 145700000:
                                                        ent_id = order_we_sendB5.iloc[monitor_order]['交易所委托号']
                                                        monitor_order += 1
                                                        log(fr'{t}：{monitor_order+1}单在145700000之前预测可以成交，撤单第{monitor_order}单')
                                                        
                                                        # print(fr'{t}：{monitor_order+1}单在145700000之前预测可以成交，撤单第{monitor_order}单')
                                                        # function_log.log_info(fr'{t}_{tk}：{monitor_order+1}单在145700000之前预测可以成交，撤单第{monitor_order}单')
        
                                                        if ent_id in ordlist:
                                                            ordlist.remove(ent_id)
                                           
     
                                                
                                # 时间大于n2，则等待成交，撤掉其余单子
                                if t >= n2_time : #and isinstance(pred_next, pd.Series) and len(pred_next) > 0:
    
                                    will_deal = consum < pred_next.iloc[0]
                                    order_no = monitor_order + 1
                                    entrust_id = pred_next.index[0]
                                    
                                    if board3 ==0:
                                        log(f"{t}_{tk}：n2时间内这单{order_no}{'会' if will_deal else '不会'}成交！")
                                        board3 = 1
                                        
                                    deal_real = deal_df_by_order_id[deal_df_by_order_id['成交代码']!='C']
                                    if deal_df_by_order_id is not None and ordlist[0] in deal_real.index:
                                        final = deal_df_by_order_id.loc[[ordlist[0]]].reset_index()
                                        if not final.empty:
                                            timedeal_list.append(final)
                                            nstock = nstock+1
                                   
                                            log(rf'第{monitor_order+1}单成交')
                                            print(final)
                                            done = True
                                            break
                                                
                                    else:
                                        # 算一下队列是否被消耗掉
                                        if etime < 145600000:
                                            continue
                                                    
                                        snap_inc_total = step_limit_buy_with_deals(
                                                state=state3,
                                                delta_orders=entrustzt_price_930_exc,
                                                delta_deals=deal_df_exc,
                                                # 如果你的委托号是整型列，设 True 性能更好：
                                                order_id_is_int=True,
                                                # U 的含义：'override' 覆盖为新量；'delta' 代表在原量基础上追加
                                                update_mode='override',
                                                emit_empty=False,        # 队列为空时是否也产出一行快照
                                                sort_ids=False           # 输出队列顺序：False 保持 FIFO；True 按编号升序
                                            )
    
                                        # fakeid = entrust_df['交易所委托号'].iloc[0]  
                                        fakeid = result_df['交易所委托号'].iloc[0]
                                        lst = snap_inc_total['委托编号列表'].iloc[0]

                                        minid = next((x for x in lst if x != fakeid), None)
                                        # minid = snap_inc_total['委托编号列表'].iloc[0][0]
                                        entrust_df = order_we_sendB5.iloc[0]
                                        # 炸板怎么办
                                        # result_df2 = compute_waiting_qty_before_entrust(snap_inc, entrust_df)
                                        # queue2 = result_df2['排队数量'].iloc[0]
                                        # if queue2>0:
                                        #     pass
                                        
                                        if fakeid < minid :
                                            print('最终成交')
                                            entrust_df['成交价格'] = ztprice
                                            log(rf'第{monitor_order+1}单成交,计算消耗队列')
                                            if isinstance(entrust_df, pd.Series):
                                                entrust_df = entrust_df.to_frame().T  # 转换为 DataFrame 并转置
                                            timedeal_list.append(entrust_df)
                                            nstock = nstock+1
                                            done=True
                                            break
                                        else:
                                            print('最终没成交')
                                            done=True
                                            break
                                                        
                                    # 成交结果统一处理
                                    # 优化：使用预索引加速查找
                                    # deal_real = deal_df_by_order_id[deal_df_by_order_id['成交代码']!='C']
                                    # if deal_real is not None and entrust_id in deal_real.index:
                                    #     final = deal_real.loc[[entrust_id]].reset_index()
                                    # # else:
                                    # #     final = pd.DataFrame()
                                    # if not final.empty:
                                    #     timedeal_list.append(final)
                                    #     nstock = nstock+1
                                    #     # done = True
                                        
                                    # if final.empty:
                                    #     log(f"{t}_{tk}：n2时间内这单{order_no}"
                                    #         f"{'会' if will_deal else '不会'}成交,最终没有成交")
                                    #     print(f"{t}_{tk}：n2时间内这单{order_no}"
                                    #         f"{'会' if will_deal else '不会'}成交,最终没有成交")

                                    #     break
                                    
                                    # elif len(final):
                                    #     # print(fr'第{monitor_order+1}单最终成交了')
                                    #     log(fr'第{monitor_order}单成交')a
                                    #     print(fr'第{monitor_order}单成交')
                                    #     done = Truea
                                    #     # function_log.log_info(
                                    #     #    fr'第{monitor_order}单成交'
                                    #     # )
                                    #     break
                              
                        
                        # 最近1000个点的消耗单子的数量 计算diffs 并计算slope和预测下一个时刻的消耗单子的数量
                        # 炸板成交 应该是预测迟钝了？
                        # 找几个看一下
                        
                        ahead_total_series = result_df['排队数量'].dropna()
                        
                        fc_state, pred_next,slope,level_star = update_and_predict_next10ms_with_carry( fc_state,
                                                                                inc_df=inc_padded,  
                                                                                ahead_total = ahead_total_series,
                                                                                index = result_df['交易所委托号'].dropna(),
                                                                                horizon_ms=1000,        
                                                                                time_col="start_time",
                                                                                value_col="总量",
                                                                                ticker_col="Ticker"
                                                                            )
                        
 
            # # ===== 优化：循环结束后，一次性合并所有timedeal =====
            if timedeal_list:
                timedeal_batch = pd.concat(timedeal_list, ignore_index=True)
                timedeal_total = pd.concat([timedeal_total, timedeal_batch], ignore_index=True)
            
            # # ===== 合并每个时间戳的记录 =====
            # if timestamp_records:
            #     # 合并所有merged_data记录（包含pred_next, slope, level_star, ahead_total）
            #     merged_data_all = pd.concat([rec['merged_data'] for rec in timestamp_records], ignore_index=True)
            #     # 合并所有final_df记录
            #     final_df_all = pd.concat([rec['final_df'] for rec in timestamp_records], ignore_index=True)
                
            #     # 保存到全局变量
            #     global pred_next_history, final_df_history
            #     if pred_next_history.empty:
            #         pred_next_history = merged_data_all
            #         final_df_history = final_df_all
            #     else:
            #         pred_next_history = pd.concat([pred_next_history, merged_data_all], ignore_index=True)
            #         final_df_history = pd.concat([final_df_history, final_df_all], ignore_index=True)
# snap_inc_total.to_excel('2016_01_04_002004.xlsx')

spd.report("ticks fetch_commissions loop") 
   
         



# 初始化一个空的列表，用来保存转换后的 DataFrame
df_list = []

for item in timedeal_list:
    # 判断元素类型是 Series
    if isinstance(item, pd.Series):
        # 将 Series 转换为 DataFrame
        df = item.to_frame().T  # 转换为 DataFrame 并转置（T）以确保每个 Series 成为一行
        df_list.append(df)
    elif isinstance(item, pd.DataFrame):
        # 如果元素已经是 DataFrame，直接添加
        df_list.append(item)

# 使用 pd.concat 拼接所有的 DataFrame
result_df = pd.concat(df_list, axis=0, ignore_index=True)

# 显示结果
print(result_df)

#%                
# time1 =time.time()                    
# print(time1- times)                
timedeal_tota1l=result_df.drop_duplicates(subset=['万得代码','自然日'])
# len(timedeal_tota1l)

dstart = timedeal_tota1l['自然日'].min()
dend = timedeal_tota1l['自然日'].max()

timedeal_tota1l.groupby(['自然日']).count()
# # trades120240201_20240301
# mask = (timedeal_tota1l['交易所代码'] == 2369) & (timedeal_tota1l['成交价格'].isna())
# timedeal_tota1l.loc[mask, '成交价格'] = timedeal_tota1l.loc[mask, '委托价格']

# dstart = 20240201
# dend = 20240301
# timedeal_tota1l= pd.read_excel(fr'C:\Users\fyx90\Desktop\excel\打板回测\bt_data\result\trades1{dstart}_{dend}.xlsx') 
timedeal_tota1l.to_excel(fr'F:\snapshot_zttime\ztpb_trades\tradesV10_{lb_window}_benchmark_{max_nstock}_n1{n1}_n2{n2}_{dstart}_{dend}.xlsx')       
          

                        #%%
timedeal2_tota1l= pd.read_excel(fr'F:\snapshot_zttime\ztpb_trades\tradesV10_300_benchmark_10_n15_n210_20240506.0_20240517.xlsx')       
timedeal3_tota1l= pd.read_excel(fr'F:\snapshot_zttime\ztpb_trades\tradesV10_300_benchmark_10_n15_n210_20240521_20240531.xlsx')       

totaltrades = pd.concat([ timedeal2_tota1l,timedeal3_tota1l ])   

dstart = totaltrades['自然日'].min()
dend = totaltrades['自然日'].max()


totaltrades.to_excel(fr'C:\Users\fyx90\Desktop\excel\打板回测\bt_data\result\tradesV10_benchmark_{max_nstock}_n1{n1}_n2{n2}_{dstart}_{dend}.xlsx')           
# sellprice = 
# total_sell_price1 = pd.read_excel(fr'F:\res\intermediate\next_day_sell_20240101_20240201.xlsx')
# total_sell_price2 = pd.read_excel(fr'F:\res\intermediate\next_day_sell_20240201_20241231.xlsx')
# total_sell_price = pd.concat([total_sell_price1,total_sell_price2])
timedeal_tota1l = pd.read_excel(fr'F:\res\intermediate\tradesV4__20240201_20240219_benchmark.xlsx')       
total_sell_price = pd.read_excel(fr'F:\res\intermediate\next_day_sell_20240201_20240301.xlsx')

totalsellprice = total_sell_price[['万得代码','buy_date','自然日','成交价']]
merged_data = pd.merge( timedeal_tota1l,totalsellprice , left_on=['万得代码','自然日'],right_on=['万得代码','buy_date'] ,how='left' )

# dstart= 20240201
# dend= 20240301
# merged_data=pd.read_excel(fr'C:\Users\fyx90\Desktop\excel\打板回测\bt_data\result\trades_{dstart}_{dend}_match.xlsx')
# merged_data.to_excel(fr'C:\Users\fyx90\Desktop\excel\打板回测\bt_data\result\trades_{dstart}_{dend}_match.xlsx')
# merged_data1 = merged_data.drop_duplicates(keep='first',subset=['tk','buy_date','t','seg_start']).dropna(subset=['成交价'])
merged_data['ret_close'] = merged_data['成交价']/merged_data['成交价格'] - 1
# daily_Ret = merged_data.groupby('buy_date')[['ret']].mean()
# nv = (daily_Ret+1).cumprod()
# nv.plot()
print(len(merged_data[merged_data['标签']=='涨停']))
print(len(merged_data[merged_data['标签']=='炸板']))


def day_ret_rule(g):
    n = len(g)
    numOfstock = 10
    if n < numOfstock:
        return 1/numOfstock * g['ret_close'].sum()  # 每只按 10% 仓位
    else:
        return g['ret_close'].mean()        # 等权平均收益
    
daily_ret = (
           merged_data.groupby('buy_date', as_index=False)
             .apply(lambda g: pd.Series({'ret': day_ret_rule(g)}))
       )    
    
nv = (daily_ret+1).cumprod()
nv['ret'].plot()


#%%

# 剔除一字板
# 剔除短期暴涨的 连续涨停板

dfnv=pd.read_excel(fr'C:\Users\fyx90\Desktop\cpp\emotion_longtou.xlsx')

trades=pd.read_excel(fr'F:\snapshot_zttime\ztpb_trades\tradesV10_300_benchmark_10_n10_n210_20240201.0_20240301.0.xlsx')

nvtiming = pd.merge( dfnv, trades, left_on=['Date'],right_on=['自然日'] ,how='right'  )

nvtiming['Ret_93500000'] = nvtiming['Ret_93500000'].fillna(0)
tradtiming = nvtiming[nvtiming['Ret_93500000']>-0.08]

tradtiming.to_excel(fr'F:\snapshot_zttime\ztpb_trades\tradesV10_300_benchmark_10_n10_n210_20240201timing.xlsx')


#%%
os.chdir(r"C:\Users\fyx_ainvest\Documents\我的坚果云\爱玩特\实习生\codes_wly\codes_wly\shortIV")
os.chdir(r"C:\Users\fyx90\Desktop\cpp\zbdx\sanbiao")
# import config
from settlement import write_sanbiao
os.chdir(r"C:\Users\fyx90\Desktop\cpp\zbdx\sanbiao")
# import sanbiao_ztpb 
from sanbiao_ztpb import *
# file = fr'totaltrade_V12_n1_0_{maxdate}_{selltime}.xlsx'
file = 'totaltradeV12_n1_0_20240301_103000000timing'+'.xlsx'
# sleep_time = 300 # 每5分钟检查一次
monitor_dir = r'F:\snapshot_zttime\ztpb_trades'
sanbiao_save_dir = r'C:\Users\fyx90\Desktop\excel\zbhf\sanbiao'
intermediate_file_dir = r'C:\Users\fyx90\Desktop\excel\zbhf\intermediate'
      
# totaltradeV10_20240201_20240701_103000000
# 原三表      totaltradeV6_benchmark_{mindate}_{maxdate}_{selltime}           


record = pd.read_excel(os.path.join(monitor_dir, file))
suffix = file.replace('.xlsx', '')
kwards = get_args(record.columns.to_list())
processed_record = get_open_close_record(record, **kwards)
# processed_record.to_excel('temp.xlsx', index=False)
processed_record = cal_trade_num_commission(processed_record, total_value=10000000)
processed_record.to_excel(os.path.join(intermediate_file_dir, f'trade_record_{suffix}.xlsx'), index=False)

# record.to_excel(os.path.join(intermediate_file_dir, f'trade_record_{suffix}.xlsx'), index=False)
write_sanbiao(os.path.join(intermediate_file_dir, f'trade_record_{suffix}.xlsx'), cap=10000000, save_path=os.path.join(sanbiao_save_dir, f'sanbiao_{suffix}.xlsx'), end=None, intraday=False, bond=False)


        
        

