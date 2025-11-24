from enum import Enum
from random import random, choice, gauss
import math
from typing import List, Dict, Optional

class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1
        return self.count

    def reset(self):
        self.count = 0

counter = Counter()

class NodeType(Enum):
    INPUT = "Input"
    OUTPUT = "Output"
    HIDDEN = "Hidden"

class Node:
    def __init__(self, ntype: NodeType, idx: int = counter()):
        self.idx = idx
        self.ntype = ntype
        self.value = 0.0
        self.activation = lambda x: math.tanh(x)  # Tanh activation (-1 to 1)

    def activate(self):
        self.value = self.activation(self.value)
        return self.value

    def __repr__(self):
        return f"Node(id={self.idx}, type={self.ntype}, value={self.value:.2f})"

    def copy(self):
        new_node = Node(self.ntype, self.idx)
        new_node.value = self.value
        return new_node

class Connection:
    def __init__(self, in_node: int, out_node: int, weight: float = None, 
                 idx: int = counter(), enabled: bool = True):
        self.idx = idx
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight if weight is not None else (random() * 2 - 1)
        self.enabled = enabled

    def forward(self, input_value: float) -> float:
        return input_value * self.weight if self.enabled else 0.0

    def copy(self):
        return Connection(self.in_node, self.out_node, self.weight, self.idx, self.enabled)

class Genome:
    def __init__(self, nodes: List[Node], connections: List[Connection]):
        self.nodes = nodes
        self.connections = connections
        self.fitness = 0.0
        self.adjusted_fitness = 0.0
        self.species = None

    def activate(self, inputs: List[float]) -> List[float]:
        # Reset non-input nodes
        for node in self.nodes:
            if node.ntype != NodeType.INPUT:
                node.value = 0.0
        
        # Set input values
        input_nodes = sorted([n for n in self.nodes if n.ntype == NodeType.INPUT], key=lambda x: x.idx)
        for i, node in enumerate(input_nodes[:len(inputs)]):
            node.value = inputs[i]
        
        # Forward propagation (with multiple passes for recurrent networks)
        for _ in range(3):
            for conn in self.connections:
                in_node = next(n for n in self.nodes if n.idx == conn.in_node)
                out_node = next(n for n in self.nodes if n.idx == conn.out_node)
                out_node.value += conn.forward(in_node.value)
            
            for node in self.nodes:
                if node.ntype != NodeType.INPUT:
                    node.activate()
        
        # Get outputs
        output_nodes = sorted([n for n in self.nodes if n.ntype == NodeType.OUTPUT], key=lambda x: x.idx)
        return [n.value for n in output_nodes]

    def mutate(self, mutation_rates: Dict[str, float] = None):
        rates = mutation_rates or {
            'weight': 0.8,
            'link': 0.05,
            'node': 0.03,
            'toggle': 0.1,
            're-enable': 0.0,
            'bias': 0.7
        }

        if random() < rates['weight']:
            self.mutate_weight(rates['weight'])
        
        if random() < rates['link']:
            self.mutate_link()
        
        if random() < rates['node']:
            self.mutate_node()
        
        if random() < rates['toggle']:
            self.mutate_toggle()
        
        if random() < rates['bias']:
            self.mutate_bias()

    def mutate_weight(self, prob: float):
        for conn in self.connections:
            if random() < prob:
                if random() < 0.1:  # 10% chance for completely new weight
                    conn.weight = random() * 4 - 2
                else:  # 90% chance for slight perturbation
                    conn.weight += gauss(0, 0.1)
                    conn.weight = max(-2, min(2, conn.weight))

    def mutate_link(self):
        available_nodes = [n.idx for n in self.nodes]
        if len(available_nodes) < 2:
            return

        in_node = choice(available_nodes)
        possible_nodes = [n for n in available_nodes if n != in_node]
        if not possible_nodes:
            # No valid node to link, so just return (skip this mutation)
            return
        out_node = choice(possible_nodes)

        # Check if connection already exists
        if any(c.in_node == in_node and c.out_node == out_node for c in self.connections):
            return

        new_conn = Connection(in_node, out_node, random() * 2 - 1)
        self.connections.append(new_conn)

    def mutate_node(self):
        if not self.connections:
            return

        conn = choice(self.connections)
        if not conn.enabled:
            return

        conn.enabled = False
        new_node = Node(NodeType.HIDDEN)

        # Create new connections
        conn1 = Connection(conn.in_node, new_node.idx, 1.0)
        conn2 = Connection(new_node.idx, conn.out_node, conn.weight)

        self.nodes.append(new_node)
        self.connections.extend([conn1, conn2])

    def mutate_toggle(self):
        if self.connections:
            conn = choice(self.connections)
            conn.enabled = not conn.enabled

    def mutate_bias(self):
        output_nodes = [n for n in self.nodes if n.ntype == NodeType.OUTPUT]
        if output_nodes and random() < 0.5:
            node = choice(output_nodes)
            node.value += gauss(0, 0.1)

    def crossover(self, partner: 'Genome') -> 'Genome':
        """Create offspring genome from two parents"""
        # Create new nodes
        child_nodes = []
        node_map = {}  # Maps parent node IDs to child node IDs

        # Add nodes from both parents
        for parent in [self, partner]:
            for node in parent.nodes:
                if node.idx not in node_map:
                    new_node = node.copy()
                    new_node.idx = counter()
                    child_nodes.append(new_node)
                    node_map[node.idx] = new_node.idx

        # Create connections
        child_conns = []
        conn_innovation = {c.idx: c for c in self.connections + partner.connections}

        for innov, conn in conn_innovation.items():
            # Randomly choose which parent's connection to take
            if conn.idx in [c.idx for c in self.connections] and conn.idx in [c.idx for c in partner.connections]:
                # Matching gene - inherit randomly
                parent_conn = choice([c for c in self.connections if c.idx == conn.idx] +
                                    [c for c in partner.connections if c.idx == conn.idx])
            elif conn.idx in [c.idx for c in self.connections]:
                # Disjoint gene from self
                parent_conn = next(c for c in self.connections if c.idx == conn.idx)
            elif conn.idx in [c.idx for c in partner.connections]:
                # Disjoint gene from partner
                parent_conn = next(c for c in partner.connections if c.idx == conn.idx)
            else:
                continue

            # Create new connection with new node IDs
            new_conn = Connection(
                node_map[parent_conn.in_node],
                node_map[parent_conn.out_node],
                parent_conn.weight,
                parent_conn.idx,
                parent_conn.enabled
            )
            child_conns.append(new_conn)

        return Genome(child_nodes, child_conns)

class Model:
    def __init__(self, input_size: int, output_size: int):
        self.input_nodes = [Node(NodeType.INPUT) for _ in range(input_size)]
        self.output_nodes = [Node(NodeType.OUTPUT) for _ in range(output_size)]
        self.genome = Genome(
            nodes=self.input_nodes + self.output_nodes,
            connections=[]
        )
        # Create initial connections
        for _ in range(input_size * output_size):
            self.genome.mutate_link()
        counter.reset()