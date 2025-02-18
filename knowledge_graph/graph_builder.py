import networkx as nx

class KnowledgeGraph:
    """Represents a knowledge graph for the codebase.
    
    Attributes:
        graph: A NetworkX graph object.
    """
    
    def __init__(self):
        self.graph = nx.Graph()
    
    def add_node(self, node_id: str, attributes: dict):
        """Add a node to the graph.
        
        Args:
            node_id: Unique identifier for the node.
            attributes: Dictionary of node attributes.
        """
        self.graph.add_node(node_id, **attributes)
    
    def add_edge(self, source_id: str, target_id: str, relation: str, attributes: dict = None):
        """Add an edge between two nodes.
        
        Args:
            source_id: Identifier for the source node.
            target_id: Identifier for the target node.
            relation: Type of relationship between the nodes.
            attributes: Dictionary of edge attributes.
        """
        if attributes is None:
            attributes = {}
        self.graph.add_edge(source_id, target_id, relation=relation, **attributes)
    
    def get_node(self, node_id: str) -> dict:
        """Get a node from the graph.
        
        Args:
            node_id: Identifier for the node.
        
        Returns:
            Dictionary of node attributes, or None if the node is not found.
        """
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]
        return None
    
    def get_neighbors(self, node_id: str) -> list:
        """Get the neighbors of a node.
        
        Args:
            node_id: Identifier for the node.
        
        Returns:
            List of neighbor node IDs.
        """
        if self.graph.has_node(node_id):
            return list(self.graph.neighbors(node_id))
        return []
    
    def search_nodes(self, query: str, attribute: str = 'name') -> list:
        """Search for nodes based on an attribute.
        
        Args:
            query: Search query string.
            attribute: Attribute to search on.
        
        Returns:
            List of node IDs that match the query.
        """
        results = []
        for node_id, attributes in self.graph.nodes(data=True):
            if query.lower() in str(attributes.get(attribute, '')).lower():
                results.append(node_id)
        return results
