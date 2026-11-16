import argparse
import json
from pathlib import Path

import yaml


def build_compose(graph):
    if graph["unresolved_arrows"]:
        raise ValueError("Resolve ambiguous arrows before generating Compose.")

    nodes = {node["id"]: node for node in graph["nodes"]}
    if not nodes or len(nodes) != len(graph["nodes"]):
        raise ValueError("The graph must contain nodes with unique IDs.")

    services = {}
    password = "${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD}"

    for node_id, node in nodes.items():
        service = {"image": node["image"]}

        if node["type"] == "service":
            if "ports" in node:
                service["ports"] = node["ports"]
        elif node["type"] == "database":
            service["environment"] = {
                "POSTGRES_DB": "appdb",
                "POSTGRES_USER": "appuser",
                "POSTGRES_PASSWORD": password,
            }
        else:
            raise ValueError(f"Unsupported symbol type: {node['type']}")

        services[node_id] = service

    for edge in graph["edges"]:
        source, target = edge["source"], edge["target"]
        if source not in nodes or target not in nodes:
            raise ValueError(f"Edge references an unknown node: {source} → {target}")
        if nodes[source]["type"] != "service" or nodes[target]["type"] != "database":
            raise ValueError(f"Expected a service-to-database edge: {source} → {target}")
        if "environment" in services[source]:
            raise ValueError(f"{source} can connect to only one database in this version.")

        database_env = services[target]["environment"]
        services[source]["environment"] = {
            "DB_HOST": target,
            "DB_PORT": "5432",
            "DB_NAME": database_env["POSTGRES_DB"],
            "DB_USER": database_env["POSTGRES_USER"],
            "DB_PASSWORD": database_env["POSTGRES_PASSWORD"],
        }

    return {"services": services}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docker-compose.yml"))
    args = parser.parse_args()

    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    compose = build_compose(graph)
    args.output.write_text(yaml.safe_dump(compose, sort_keys=False), encoding="utf-8")
    print(f"Generated {args.output}")


if __name__ == "__main__":
    main()