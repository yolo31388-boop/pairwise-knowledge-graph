"""
游戏内知识图谱与推理系统 - 核心模块

功能：
- build_graph(): 知识图谱构建（实体 + 关系）
- reason(): 基于规则的前向链推理，带解释
- ontology(): 本体定义（类层次 + 属性约束）
- semantic_network(): 语义网络（激活传播 + 相似度）
- query(): 模式匹配查询（带索引优化）
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time


@dataclass
class Config:
    pass


def _is_var(term) -> bool:
    return isinstance(term, str) and term.startswith("?")


class KnowledgeGraph:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        # 知识图谱：实体与关系
        self._entities: Dict[str, Dict] = {}
        self._triples: List[Tuple[str, str, str]] = []
        self._triple_set = set()
        # 查询优化：三个维度上的索引
        self._index_s: Dict[str, set] = {}
        self._index_p: Dict[str, set] = {}
        self._index_o: Dict[str, set] = {}
        # 推理：规则与派生事实的解释
        self._rules: List[Dict] = []
        self._explanations: Dict[Tuple, Dict] = {}
        # 本体：类层次与属性约束
        self._classes: Dict[str, Optional[str]] = {}
        self._properties: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # 知识图谱构建
    # ------------------------------------------------------------------
    def add_entity(self, name: str, entity_type: Optional[str] = None, **attrs):
        entity = self._entities.setdefault(name, {"type": None, "attrs": {}})
        if entity_type is not None:
            entity["type"] = entity_type
        entity["attrs"].update(attrs)
        return entity

    def add_relation(self, subject: str, predicate: str, obj: str):
        triple = (subject, predicate, obj)
        if triple in self._triple_set:
            return triple
        for term in (subject, obj):
            if term not in self._entities:
                self.add_entity(term)
        idx = len(self._triples)
        self._triples.append(triple)
        self._triple_set.add(triple)
        self._index_s.setdefault(subject, set()).add(idx)
        self._index_p.setdefault(predicate, set()).add(idx)
        self._index_o.setdefault(obj, set()).add(idx)
        return triple

    def build_graph(self, entities: Optional[List] = None,
                    relations: Optional[List[Tuple]] = None) -> Dict:
        """批量构建知识图谱，返回构建统计。"""
        for entity in entities or []:
            if isinstance(entity, str):
                self.add_entity(entity)
            elif isinstance(entity, dict):
                name = entity.get("name")
                attrs = {k: v for k, v in entity.items() if k not in ("name", "type")}
                self.add_entity(name, entity.get("type"), **attrs)
            else:
                name, etype = entity[0], entity[1] if len(entity) > 1 else None
                self.add_entity(name, etype)
        for relation in relations or []:
            subject, predicate, obj = relation
            self.add_relation(subject, predicate, obj)
        return {
            "entities": len(self._entities),
            "relations": len(self._triples),
        }

    def entities(self) -> List[str]:
        return sorted(self._entities)

    def relations(self) -> List[Tuple[str, str, str]]:
        return list(self._triples)

    def get_entity(self, name: str) -> Optional[Dict]:
        return self._entities.get(name)

    # ------------------------------------------------------------------
    # 本体：类层次与属性约束
    # ------------------------------------------------------------------
    def define_class(self, name: str, parent: Optional[str] = None):
        if parent is not None and parent not in self._classes:
            self._classes[parent] = None
        self._classes[name] = parent
        return name

    def define_property(self, name: str, domain: Optional[str] = None,
                        range=None):
        """定义属性约束：domain 为主语类，range 为类名 / Python类型 / 允许值集合。"""
        self._properties[name] = {"domain": domain, "range": range}
        return name

    def ontology(self) -> Dict:
        return {
            "classes": {name: parent for name, parent in self._classes.items()},
            "properties": {name: dict(spec) for name, spec in self._properties.items()},
        }

    def ancestors(self, name: str) -> List[str]:
        chain, seen, current = [], set(), name
        while current in self._classes and self._classes[current] is not None:
            current = self._classes[current]
            if current in seen:
                break
            seen.add(current)
            chain.append(current)
        return chain

    def is_subclass(self, child: str, parent: str) -> bool:
        return child == parent or parent in self.ancestors(child)

    def check_constraint(self, prop: str, subject: str, value) -> bool:
        spec = self._properties.get(prop)
        if spec is None:
            return True
        domain, rng = spec.get("domain"), spec.get("range")
        if domain is not None:
            entity = self._entities.get(subject)
            etype = entity.get("type") if entity else None
            if etype is None or not self.is_subclass(etype, domain):
                return False
        if rng is None:
            return True
        if isinstance(rng, type):
            return isinstance(value, rng)
        if isinstance(rng, (set, frozenset, list, tuple)):
            return value in rng
        # rng 为类名：值必须是该类的实体，或（无类型信息时）已知实体
        entity = self._entities.get(value)
        if entity is None:
            return False
        etype = entity.get("type")
        return etype is not None and self.is_subclass(etype, rng)

    def validate(self) -> List[Tuple]:
        """返回所有违反本体约束的三元组。"""
        violations = []
        for subject, predicate, obj in self._triples:
            if not self.check_constraint(predicate, subject, obj):
                violations.append((subject, predicate, obj))
        return violations

    # ------------------------------------------------------------------
    # 推理：规则 + 前向链 + 解释
    # ------------------------------------------------------------------
    def add_rule(self, name: str, conditions: List[Tuple], conclusion: Tuple):
        """conditions/conclusion 为三元组模式，? 开头为变量。"""
        self._rules.append({
            "name": name,
            "conditions": list(conditions),
            "conclusion": tuple(conclusion),
        })
        return name

    def _match_pattern(self, pattern: Tuple, candidate_pool=None):
        """对单个模式在候选三元组集合上做匹配，产出绑定列表。"""
        s, p, o = pattern
        if candidate_pool is None:
            candidate_pool = range(len(self._triples))
        bindings = []
        for idx in candidate_pool:
            ts, tp, to = self._triples[idx]
            binding = {}
            for pattern_term, actual in ((s, ts), (p, tp), (o, to)):
                if _is_var(pattern_term):
                    binding[pattern_term] = actual
                elif pattern_term != actual:
                    break
            else:
                bindings.append((idx, binding))
        return bindings

    def _candidates_for(self, pattern: Tuple, binding: Dict):
        """查询优化：按已绑定的位置选择最有选择性的索引。"""
        s, p, o = (binding.get(t, t) for t in pattern)
        pools = []
        if not _is_var(s):
            pools.append(self._index_s.get(s, set()))
        if not _is_var(p):
            pools.append(self._index_p.get(p, set()))
        if not _is_var(o):
            pools.append(self._index_o.get(o, set()))
        if not pools:
            return range(len(self._triples))
        return min(pools, key=len)

    def _join(self, patterns: List[Tuple]) -> List[Dict]:
        """多模式连接查询，按选择性排序以减少中间结果。"""
        ordered = sorted(
            patterns,
            key=lambda pat: sum(0 if _is_var(t) else 1 for t in pat),
            reverse=True,
        )
        results: List[Dict] = [{}]
        for pattern in ordered:
            next_results = []
            for binding in results:
                grounded = tuple(binding.get(t, t) for t in pattern)
                pool = self._candidates_for(grounded, binding)
                for _, match in self._match_pattern(grounded, pool):
                    merged = dict(binding)
                    conflict = False
                    for var, val in match.items():
                        if var in merged and merged[var] != val:
                            conflict = True
                            break
                        merged[var] = val
                    if not conflict:
                        next_results.append(merged)
            results = next_results
        return results

    def reason(self, max_rounds: int = 100) -> Dict:
        """前向链推理：反复应用规则推导新知识，并记录每步解释。"""
        derived_all: List[Tuple] = []
        for _ in range(max_rounds):
            new_facts = []
            for rule in self._rules:
                for binding in self._join(rule["conditions"]):
                    conclusion = tuple(binding.get(t, t) for t in rule["conclusion"])
                    if any(_is_var(t) for t in conclusion):
                        continue
                    if conclusion in self._triple_set:
                        continue
                    premises = [tuple(binding.get(t, t) for t in c)
                                for c in rule["conditions"]]
                    self.add_relation(*conclusion)
                    self._explanations[conclusion] = {
                        "rule": rule["name"],
                        "premises": premises,
                    }
                    new_facts.append(conclusion)
            derived_all.extend(new_facts)
            if not new_facts:
                break
        self._history.append({"derived": len(derived_all), "time": time.time()})
        return {
            "derived": derived_all,
            "explanations": {t: dict(self._explanations[t]) for t in derived_all},
        }

    def explain(self, triple: Tuple) -> List[Dict]:
        """递归展开一个结论的完整推理链。"""
        chain = []
        seen = set()

        def expand(fact):
            if fact in seen:
                return
            seen.add(fact)
            info = self._explanations.get(fact)
            if info is None:
                return
            for premise in info["premises"]:
                expand(premise)
            chain.append({"rule": info["rule"], "conclusion": fact,
                          "premises": list(info["premises"])})

        expand(tuple(triple))
        return chain

    # ------------------------------------------------------------------
    # 语义网络：激活传播 + 相似度
    # ------------------------------------------------------------------
    def _neighbors(self, concept: str):
        for idx in self._index_s.get(concept, ()):
            _, p, o = self._triples[idx]
            yield o, p
        for idx in self._index_o.get(concept, ()):
            s, p, _ = self._triples[idx]
            yield s, p

    def propagate(self, start: str, activation: float = 1.0,
                  decay: float = 0.5, max_depth: int = 3) -> Dict[str, float]:
        """从 start 概念沿关系向外传播激活，逐层衰减。"""
        levels = {start: activation}
        frontier = {start: activation}
        for _ in range(max_depth):
            next_frontier = {}
            for concept, value in frontier.items():
                for neighbor, _ in self._neighbors(concept):
                    spread = value * decay
                    if spread > levels.get(neighbor, 0.0):
                        next_frontier[neighbor] = max(
                            spread, next_frontier.get(neighbor, 0.0))
            for concept, value in next_frontier.items():
                if value > levels.get(concept, 0.0):
                    levels[concept] = value
            frontier = {c: v for c, v in next_frontier.items()
                        if levels.get(c) == v and c != start}
            if not frontier:
                break
        return levels

    def similarity(self, a: str, b: str) -> float:
        """基于共享关系签名（谓词+邻居）的 Jaccard 相似度。"""
        if a == b:
            return 1.0
        sig_a = {(p, n) for n, p in self._neighbors(a)}
        sig_b = {(p, n) for n, p in self._neighbors(b)}
        if not sig_a and not sig_b:
            return 0.0
        union = sig_a | sig_b
        if not union:
            return 0.0
        return len(sig_a & sig_b) / len(union)

    # ------------------------------------------------------------------
    # 查询：模式匹配 + 索引优化
    # ------------------------------------------------------------------
    def query(self, *patterns: Tuple) -> List[Dict]:
        """按一个或多个三元组模式查询，返回变量绑定列表。"""
        if not patterns:
            return []
        return self._join(list(patterns))

    def query_triples(self, subject=None, predicate=None, obj=None) -> List[Tuple]:
        """便捷查询：None 视为通配，返回匹配的三元组。"""
        pattern = (subject if subject is not None else "?s",
                   predicate if predicate is not None else "?p",
                   obj if obj is not None else "?o")
        return [tuple(b.get(t, t) for t in pattern) for b in self.query(pattern)]

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []
        self._entities = {}
        self._triples = []
        self._triple_set = set()
        self._index_s = {}
        self._index_p = {}
        self._index_o = {}
        self._rules = []
        self._explanations = {}
        self._classes = {}
        self._properties = {}
