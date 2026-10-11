"""Update P0B ablation slides in-place within an OOXML copy, preserving other slides."""
from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}
ROOT = Path(__file__).resolve().parents[1]


def tag(name):
    prefix, local = name.split(":")
    return "{" + NS[prefix] + "}" + local


def text_of(node):
    return "\n".join(t.text or "" for t in node.iter(tag("a:t")))


def write_text(container, value):
    paragraphs = container.findall("a:p", NS)
    if not paragraphs:
        raise ValueError("Text container has no paragraph")
    template = copy.deepcopy(paragraphs[0])
    for p in paragraphs:
        container.remove(p)
    for line in value.split("\n"):
        p = copy.deepcopy(template)
        runs = p.findall("a:r", NS)
        r = copy.deepcopy(runs[0]) if runs else ET.Element(tag("a:r"))
        for child in list(p):
            if child.tag in (tag("a:r"), tag("a:br"), tag("a:fld")):
                p.remove(child)
        for child in list(r):
            if child.tag == tag("a:t"):
                r.remove(child)
        ET.SubElement(r, tag("a:t")).text = line
        end = p.find("a:endParaRPr", NS)
        p.insert(list(p).index(end) if end is not None else len(p), r)
        container.append(p)


def update_slide(data):
    for _, pair in ET.iterparse(io.BytesIO(data), events=("start-ns",)):
        prefix, uri = pair
        if not prefix.startswith("ns"):
            ET.register_namespace(prefix, uri)
    root = ET.fromstring(data)
    replacements = {
        "研究一：耦合、路由与分支的独立贡献": "研究一：P0B基线与模块消融设置",
        "MOSI · p0B 结构 · 完整模型 + 10 项消融 · 部分运行批大小调整，待分组核验":
            "MOSI · P0B历史基线 · 6组×3种子 · batch=4 / lr=4e-4 / dropout=0.6",
        "依据：9/19 基准对齐审计与消融队列；展示形式参考 TF-Mamba（EMNLP Findings 2025）Table 5。":
            "依据：P0B最优日志（2026-09-18）与 ablation_plan.json；新消融结果待重训。",
        "组件对照｜每行对应一个独立实验": "统一协议｜完整基线 + 5项模块移除",
        "01  耦合与路由": "01  固定评测协议",
        "自身写入 / K / V 分别对照\n区分来源交互与路由权重":
            "种子：42 / 1111 / 2024\n验证集Acc2选模；报均值±标准差",
        "02  分支与门控": "02  有效模块干预",
        "GLCE 整体及全局 / 局部\nGCMN 与 BSSM 独立移除":
            "门控分别关闭；大模块整体移除\n特征、批大小与选模规则固定",
        "03  当前进度": "03  新协议进度",
        "11/11 冒烟通过\n已完成 15/33 项；重复实验继续":
            "6组配置已设置，共18次训练\n新协议尚未完成训练与汇总",
        "其余消融结果待协议核验后汇总。历史上下文为样本行偏移，可能跨视频。":
            "历史参考：seed2024 Acc2=91.01%，Acc7=52.19%（单次结果，非本表均值）。\n去上下文含额外融合路径移除；去ISM与去跨模态均为整体结构消融。",
        "先比较耦合与路由，再拆解模态内分支；当前尚不能断言哪一模块带来最大增益。":
            "同一模型权重同时报告Acc2与Acc7；旧协议运行单独归档，不计入本表统计。",
    }
    hits = set()
    for shape in root.iter(tag("p:sp")):
        old = text_of(shape)
        if old in replacements:
            body = shape.find("p:txBody", NS)
            write_text(body, replacements[old])
            hits.add(old)
    if hits != set(replacements):
        raise ValueError(f"Slide text differs from expected template: {set(replacements) - hits}")
    shape_tree = root.find("p:cSld/p:spTree", NS)
    for shape in list(shape_tree):
        identity = shape.find("p:nvCxnSpPr/p:cNvPr", NS)
        if identity is None:
            identity = shape.find("p:nvSpPr/p:cNvPr", NS)
        if identity is not None and identity.get("name") in ("Connector 19", "Connector 20"):
            shape_tree.remove(shape)
    tables = list(root.iter(tag("a:tbl")))
    if len(tables) != 1:
        raise ValueError("Expected one ablation table")
    table = tables[0]
    old_rows = table.findall("a:tr", NS)
    total_height = sum(int(row.attrib["h"]) for row in old_rows)
    header, full, body = (copy.deepcopy(old_rows[i]) for i in (0, 1, 2))
    for row in old_rows:
        table.remove(row)
    rows = [["编号", "模型变体", "Acc2 Non0 ↑", "Acc7 ↑", "状态"],
            ["00", "完整P0B（重训基线）", "—", "—", "待运行"],
            ["01", "w/o 上下文管道", "—", "—", "待运行"],
            ["02", "w/o BSSM门控", "—", "—", "待运行"],
            ["03", "w/o GCMN门控", "—", "—", "待运行"],
            ["04", "w/o ISM整体", "—", "—", "待运行"],
            ["05", "w/o 跨模态融合", "—", "—", "待运行"]]
    for index, values in enumerate(rows):
        row = copy.deepcopy(header if index == 0 else full if index == 1 else body)
        row.set("h", str(total_height // len(rows)))
        for cell, text in zip(row.findall("a:tc", NS), values):
            write_text(cell.find("a:txBody", NS), text)
        table.append(row)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def update_deck(source, destination):
    changes = {}
    with ZipFile(source) as zin:
        changes["ppt/slides/slide13.xml"] = update_slide(zin.read("ppt/slides/slide13.xml"))
        key = "ppt/slides/slide25.xml"
        data = zin.read(key)
        old = "消融已启动，待核验运行协议分组。"
        new = "P0B消融重新设置，待三种子统一验证。"
        if old.encode() not in data:
            raise ValueError("Summary slide does not match template")
        changes[key] = data.replace(old.encode(), new.encode())
        # Replace any legacy speaker notes for the ablation page.
        key = "ppt/notesSlides/notesSlide13.xml"
        if key in zin.namelist():
            tree = ET.fromstring(zin.read(key))
            for shape in tree.iter(tag("p:sp")):
                placeholder = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
                if placeholder is not None and placeholder.get("type") == "body":
                    body = shape.find("p:txBody", NS)
                    if body is not None and body.findall("a:p", NS):
                        write_text(body, "P0B重建基线：6组×3种子，固定batch4/lr4e-4/dropout0.6。"
                                   "按验证集Acc2选模，统一报告同一checkpoint的测试Acc2/Acc7。"
                                   "历史91.01/52.19为seed2024单次参考，不能作为新三种子均值。"
                                   "去上下文同时移除额外融合；去ISM和去跨模态为整体结构消融。"
                                   "新消融尚未完成。")
            changes[key] = ET.tostring(tree, encoding="utf-8", xml_declaration=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(destination, "w") as zout:
            for info in zin.infolist():
                zout.writestr(info, changes.get(info.filename, zin.read(info.filename)))
    with ZipFile(destination) as z:
        assert z.testzip() is None
        for name in changes:
            ET.fromstring(z.read(name))
    return dict(source=str(source), output=str(destination), changed_parts=list(changes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(update_deck(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
