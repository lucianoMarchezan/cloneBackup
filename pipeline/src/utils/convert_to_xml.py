from itertools import combinations
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from src.config import *

def json_to_xml(json_file=FINAL_DATASET, xml_file=FINAL_DATASET_RQ2): 

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    root = ET.Element("clones")

    for entry in data:
        entry_id = entry.get("id", "unknown")
        clones = entry.get("clones", [])

        # Generate all unique pairs of clones
        for clone1, clone2 in combinations(clones, 2):
            c1_id = clone1.get("clone_id", "unknown")
            c2_id = clone2.get("clone_id", "unknown")

            c1_code = clone1.get("code", "")
            c2_code = clone2.get("code", "")

            # <clone> element
            clone_elem = ET.SubElement(root, "clone")

            # First fragment
            ET.SubElement(
                clone_elem,
                "source",
                file=f"{entry_id}_{c1_id}",
                startline="0",
                endline="0"
            )
            code1_elem = ET.SubElement(clone_elem, "code")
            code1_elem.text = "\n".join(c1_code.splitlines())

            # Second fragment
            ET.SubElement(
                clone_elem,
                "source",
                file=f"{entry_id}_{c2_id}",
                startline="0",
                endline="0"
            )
            code2_elem = ET.SubElement(clone_elem, "code")
            code2_elem.text = "\n".join(c2_code.splitlines())

    # Pretty print XML
    xml_str = ET.tostring(root, encoding="utf-8")
    parsed_xml = minidom.parseString(xml_str)
    pretty_xml = parsed_xml.toprettyxml(indent="    ")

    # Save to file
    with open(xml_file, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    print(f"XML saved to {xml_file}")
