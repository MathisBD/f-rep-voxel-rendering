#include "utils/dot_graph.h"
#include <assert.h>


DotGraph::DotGraph(bool directed) : m_directed(directed) {}


int DotGraph::AddNode(const std::string& label) 
{
    int id = m_nodes.size();
    m_nodes.push_back({ id, label });
    return id;
}

void DotGraph::AddEdge(int nodeFrom, int nodeTo) 
{
    m_edges.push_back({ nodeFrom, nodeTo, "" });    
}

void DotGraph::AddEdge(int nodeFrom, int nodeTo, const std::string& label) 
{
    m_edges.push_back({ nodeFrom, nodeTo, label });
}

void DotGraph::Merge(const DotGraph& other) 
{
    assert(other.m_directed == m_directed);

    int ofs = m_nodes.size();
    for (const Node& n : other.m_nodes) {
        m_nodes.push_back({ ofs + n.id, n.label });
    }    
    for (const Edge& e : other.m_edges) {
        m_edges.push_back({ ofs + e.nodeFrom, ofs + e.nodeTo, e.label });
    }
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
        sprintf(buf, "\t%d [ label=\"%s\" ]\n", node.id, node.label.c_str());
        str += std::string(buf);
    }

    // Add edges
    for (auto& edge : m_edges) {
        char buf[256];
        const char* arrow = m_directed ? "->" : "--";
        if (edge.label == "") {
            sprintf(buf, "\t%d %s %d\n", 
                edge.nodeFrom, arrow, edge.nodeTo);
        }
        else {
            sprintf(buf, "\t%d %s %d [ label=\"%s\" ]\n", 
                edge.nodeFrom, arrow, edge.nodeTo, edge.label.c_str());
        }
        str += std::string(buf);
    }

    str += "}\n";
    return str;
}