"""
游戏内知识图谱与推理系统 - 验收测试
运行方式：python -m pytest tests/test_knowledge_graph.py -q
共 15 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge.graph import KnowledgeGraph


def make_game_graph():
    """构建一个典型的游戏知识图谱：NPC、物品、地点及其关系。"""
    kg = KnowledgeGraph()
    kg.build_graph(
        entities=[
            {"name": "铁匠", "type": "NPC", "profession": "锻造"},
            {"name": "村长", "type": "NPC"},
            {"name": "勇者", "type": "NPC"},
            {"name": "铁剑", "type": "武器"},
            {"name": "铁锭", "type": "材料"},
            {"name": "村庄", "type": "地点"},
        ],
        relations=[
            ("铁匠", "出售", "铁剑"),
            ("铁剑", "由材料制成", "铁锭"),
            ("铁匠", "位于", "村庄"),
            ("村长", "位于", "村庄"),
            ("勇者", "想购买", "铁剑"),
        ],
    )
    return kg


class TestKnowledgeGraph:
    def setup_method(self):
        self.system = KnowledgeGraph()

    # ---- 知识图谱构建：实体与关系 ----
    def test_case_01(self):
        kg = make_game_graph()
        stats = kg.build_graph()
        assert stats["entities"] == 6
        assert stats["relations"] == 5
        assert set(kg.entities()) == {"铁匠", "村长", "勇者", "铁剑", "铁锭", "村庄"}

    def test_case_02(self):
        kg = make_game_graph()
        rels = kg.relations()
        assert ("铁匠", "出售", "铁剑") in rels
        assert ("铁剑", "由材料制成", "铁锭") in rels
        assert len(rels) == 5

    def test_case_03(self):
        kg = make_game_graph()
        smith = kg.get_entity("铁匠")
        assert smith["type"] == "NPC"
        assert smith["attrs"]["profession"] == "锻造"
        # 重复添加同一关系不应产生重复三元组
        kg.add_relation("铁匠", "出售", "铁剑")
        assert len(kg.relations()) == 5

    # ---- 本体：类层次与属性约束 ----
    def test_case_04(self):
        kg = self.system
        kg.define_class("NPC", parent="角色")
        kg.define_class("商人", parent="NPC")
        kg.define_class("铁匠类", parent="商人")
        ontology = kg.ontology()
        assert ontology["classes"]["商人"] == "NPC"
        assert kg.is_subclass("铁匠类", "角色")  # 传递继承
        assert kg.is_subclass("商人", "NPC")
        assert not kg.is_subclass("角色", "NPC")
        assert kg.ancestors("铁匠类") == ["商人", "NPC", "角色"]

    def test_case_05(self):
        kg = self.system
        kg.define_class("NPC")
        kg.define_class("物品")
        kg.add_entity("铁匠", "NPC")
        kg.add_entity("铁剑", "物品")
        kg.add_entity("村庄")  # 无类型
        kg.define_property("出售", domain="NPC", range="物品")
        assert kg.check_constraint("出售", "铁匠", "铁剑")
        assert not kg.check_constraint("出售", "铁剑", "铁剑")  # 主语不是NPC
        assert not kg.check_constraint("出售", "铁匠", "村庄")  # 值不在range内
        kg.add_relation("铁匠", "出售", "铁剑")
        kg.add_relation("铁剑", "出售", "铁剑")
        violations = kg.validate()
        assert violations == [("铁剑", "出售", "铁剑")]

    # ---- 推理：规则、推导与解释 ----
    def test_case_06(self):
        kg = make_game_graph()
        kg.add_rule(
            "可购买规则",
            [("?npc", "出售", "?item"), ("?hero", "想购买", "?item")],
            ("?hero", "可以买到", "?item"),
        )
        result = kg.reason()
        assert ("勇者", "可以买到", "铁剑") in result["derived"]

    def test_case_07(self):
        kg = make_game_graph()
        # 多步链式推理：位于具有传递性
        kg.add_rule("同村规则",
                    [("?a", "位于", "?place"), ("?b", "位于", "?place")],
                    ("?a", "认识", "?b"))
        kg.add_rule("熟人规则",
                    [("?a", "认识", "?b")],
                    ("?b", "认识", "?a"))
        result = kg.reason()
        assert ("铁匠", "认识", "村长") in result["derived"]
        assert ("村长", "认识", "铁匠") in result["derived"]  # 二阶推导

    def test_case_08(self):
        kg = make_game_graph()
        kg.add_rule(
            "可购买规则",
            [("?npc", "出售", "?item"), ("?hero", "想购买", "?item")],
            ("?hero", "可以买到", "?item"),
        )
        result = kg.reason()
        fact = ("勇者", "可以买到", "铁剑")
        explanation = result["explanations"][fact]
        assert explanation["rule"] == "可购买规则"
        assert ("铁匠", "出售", "铁剑") in explanation["premises"]
        # 完整推理链
        chain = kg.explain(fact)
        assert chain[-1]["conclusion"] == fact

    # ---- 查询：模式匹配与索引优化 ----
    def test_case_09(self):
        kg = make_game_graph()
        # 单模式变量查询：谁出售什么？
        results = kg.query(("?npc", "出售", "?item"))
        assert results == [{"?npc": "铁匠", "?item": "铁剑"}]
        # 常量 + 变量混合
        results = kg.query(("?who", "位于", "村庄"))
        assert {r["?who"] for r in results} == {"铁匠", "村长"}

    def test_case_10(self):
        kg = make_game_graph()
        # 多模式连接查询：找出出售勇者想买的东西的NPC
        results = kg.query(
            ("?npc", "出售", "?item"),
            ("?hero", "想购买", "?item"),
        )
        assert results == [{"?npc": "铁匠", "?item": "铁剑", "?hero": "勇者"}]

    def test_case_11(self):
        kg = make_game_graph()
        # 索引已建立：三个维度都能命中
        assert "铁匠" in kg._index_s
        assert "出售" in kg._index_p
        assert "铁剑" in kg._index_o
        # 绑定主语的查询只扫描该主语的候选集，而不是全图
        pool = kg._candidates_for(("铁匠", "?p", "?o"), {})
        assert len(pool) == 2  # 铁匠只有2条出边，而非全部5条
        assert kg.query_triples(predicate="位于") == [
            ("铁匠", "位于", "村庄"), ("村长", "位于", "村庄")]

    # ---- 语义网络：激活传播与相似度 ----
    def test_case_12(self):
        kg = make_game_graph()
        levels = kg.propagate("铁匠", activation=1.0, decay=0.5, max_depth=2)
        assert levels["铁匠"] == 1.0
        assert levels["铁剑"] == 0.5          # 一跳衰减
        assert levels["铁锭"] == 0.25         # 两跳衰减
        assert "村长" not in levels or levels["村长"] <= 0.25

    def test_case_13(self):
        kg = self.system
        kg.build_graph(relations=[
            ("猫", "是一种", "动物"), ("猫", "会", "喵喵叫"),
            ("狗", "是一种", "动物"), ("狗", "会", "汪汪叫"),
            ("汽车", "是一种", "机器"),
        ])
        sim_related = kg.similarity("猫", "狗")
        sim_unrelated = kg.similarity("猫", "汽车")
        assert 0.0 < sim_related < 1.0
        assert sim_related > sim_unrelated
        assert kg.similarity("猫", "猫") == 1.0

    # ---- 生命周期与端到端场景 ----
    def test_case_14(self):
        kg = make_game_graph()
        kg.define_class("NPC")
        kg.add_rule("r", [("?a", "位于", "?b")], ("?a", "存在于", "?b"))
        kg.reason()
        assert len(kg.relations()) > 5
        kg.reset()
        assert kg.entities() == []
        assert kg.relations() == []
        assert kg.ontology()["classes"] == {}
        assert kg.reason()["derived"] == []

    def test_case_15(self):
        """端到端：构建 -> 本体 -> 推理 -> 查询 的完整流程。"""
        kg = make_game_graph()
        kg.define_class("NPC")
        kg.define_class("武器", parent="物品")
        kg.define_property("出售", domain="NPC", range="物品")
        kg.add_entity("铁剑", "武器")
        assert kg.check_constraint("出售", "铁匠", "铁剑")
        kg.add_rule(
            "材料溯源",
            [("?npc", "出售", "?item"), ("?item", "由材料制成", "?mat")],
            ("?npc", "掌握材料", "?mat"),
        )
        result = kg.reason()
        assert ("铁匠", "掌握材料", "铁锭") in result["derived"]
        # 查询：铁匠掌握哪些材料？
        rows = kg.query(("铁匠", "掌握材料", "?mat"))
        assert [r["?mat"] for r in rows] == ["铁锭"]
        # 解释为什么铁匠掌握铁锭
        chain = kg.explain(("铁匠", "掌握材料", "铁锭"))
        assert chain[0]["rule"] == "材料溯源"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
