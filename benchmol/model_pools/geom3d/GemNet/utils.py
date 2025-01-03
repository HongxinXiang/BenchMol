import json
import os.path


def read_json(path):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    if not os.path.exists(path):
        print(f"{path} is not existing, using default scaling_factors.json")
        return {
            "comment": "GemNet",
            "QuadInteraction_1_had_rbf": 3.838575780391693,
            "QuadInteraction_1_had_cbf": 30.91912269592285,
            "QuadInteraction_1_sum_sbf": 2.015521287918091,
            "TripInteraction_1_had_rbf": 2.9607054591178894,
            "TripInteraction_1_sum_cbf": 5.57607889175415,
            "AtomUpdate_1_sum": 1.0634181648492813,
            "QuadInteraction_2_had_rbf": 3.4999656677246094,
            "QuadInteraction_2_had_cbf": 29.30245018005371,
            "QuadInteraction_2_sum_sbf": 1.9548791646957397,
            "TripInteraction_2_had_rbf": 3.0770468711853027,
            "TripInteraction_2_sum_cbf": 6.400703430175781,
            "AtomUpdate_2_sum": 1.023792326450348,
            "QuadInteraction_3_had_rbf": 3.506244122982025,
            "QuadInteraction_3_had_cbf": 30.250303268432617,
            "QuadInteraction_3_sum_sbf": 1.9496761560440063,
            "TripInteraction_3_had_rbf": 3.4999406337738037,
            "TripInteraction_3_sum_cbf": 5.825993537902832,
            "AtomUpdate_3_sum": 0.8776205033063889,
            "QuadInteraction_4_had_rbf": 3.420105278491974,
            "QuadInteraction_4_had_cbf": 30.560321807861328,
            "QuadInteraction_4_sum_sbf": 2.013889789581299,
            "TripInteraction_4_had_rbf": 3.34897518157959,
            "TripInteraction_4_sum_cbf": 5.816178321838379,
            "AtomUpdate_4_sum": 0.8766722679138184,
            "OutBlock_0_sum": 1.1001640558242798,
            "OutBlock_0_had": 3.786764442920685,
            "OutBlock_1_sum": 0.989106222987175,
            "OutBlock_1_had": 2.9567965865135193,
            "OutBlock_2_sum": 0.9261481463909149,
            "OutBlock_2_had": 2.9033637046813965,
            "OutBlock_3_sum": 0.8048739284276962,
            "OutBlock_3_had": 2.95436292886734,
            "OutBlock_4_sum": 0.8166412264108658,
            "OutBlock_4_had": 3.0642566084861755
        }

    with open(path, "r") as f:
        content = json.load(f)
    return content


def update_json(path, data):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    content = read_json(path)
    content.update(data)
    write_json(path, content)


def write_json(path, data):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_value_json(path, key):
    """ """
    content = read_json(path)

    if key in content.keys():
        return content[key]
    else:
        return None
