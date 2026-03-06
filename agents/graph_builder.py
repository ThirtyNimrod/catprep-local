import json
import networkx as nx
import matplotlib.pyplot as plt
import os
from core.logger import get_logger

logger = get_logger("graph_builder")

def build_knowledge_graph(json_file_path, output_image="graph.png"):
    """
    Reads structured Entities and Relationships from a JSON file
    and builds a Knowledge Graph using NetworkX.
    """
    if not os.path.exists(json_file_path):
        logger.error(f"File not found: {json_file_path}")
        return
        
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    G = nx.DiGraph()
    
    # Add Entities (Nodes)
    logger.info("Adding Entities...")
    for entity in data.get('entities', []):
        G.add_node(entity['id'], label=entity['name'], type=entity['type'])
        
    # Add Relationships (Edges)
    logger.info("Adding Relationships...")
    for rel in data.get('relationships', []):
        if G.has_node(rel['from']) and G.has_node(rel['to']):
            G.add_edge(rel['from'], rel['to'], relation=rel['type'])
        else:
            logger.warning(f"Missing node(s) for relationship {rel['from']} -> {rel['to']}")

    logger.info(f"Graph built successfully with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Visualize the Graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5)
    
    # Draw nodes based on type
    color_map = []
    for node, attr in G.nodes(data=True):
        t = attr.get('type', '')
        if t == 'Topic': color_map.append('lightblue')
        elif t == 'Formula': color_map.append('lightgreen')
        elif t == 'Question': color_map.append('yellow')
        else: color_map.append('lightgray')

    nx.draw(G, pos, with_labels=False, node_color=color_map, node_size=2000, 
            font_size=10, font_weight='bold', arrows=True)
            
    # Node Labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=9)
    
    # Edge Labels
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("CAT Prep Knowledge Graph Fragment")
    plt.savefig(output_image)
    logger.info(f"Saved graph visualization to {output_image}")
    return G

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="test_extraction.json", help="JSON file with entities/relations")
    args = parser.parse_args()
    build_knowledge_graph(args.file)
