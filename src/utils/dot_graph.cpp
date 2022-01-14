#include "utils/dot_graph.h"



DotGraph::DotGraph(bool directed) : m_directed(directed) {}


int DotGraph::AddNode(const std::string& label) 
{
    int id = m_nodes.size();
    m_nodes.push_back({ id, label });
    return id;
}

void DotGraph::AddEdge(int nodeFrom, int nodeTo) 
{
    m_edges.push_back({ nodeFrom, nodeTo });    
}

std::string DotGraph::Build() 
{
    std::string str = "";
    if (m_directed) {
        str += "digraph {\n";
    }    
    else {
        str += "graph {\n";
    }

    // Add nodes
    for (auto& node : m_nodes) {
        char buf[node.label.size() + 256];
        sprintf(buf, "\t%d [ label=\"%s\"]\n", node.id, node.label.c_str());
        str += std::string(buf);
    }

    // Add edges
    for (auto& edge : m_edges) {
        char buf[256];
        sprintf(buf, "\t%d %s %d\n", edge.nodeFrom, m_directed ? "->" : "--", edge.nodeTo);
        str += std::string(buf);
    }

    str += "}\n";
    return str;
}