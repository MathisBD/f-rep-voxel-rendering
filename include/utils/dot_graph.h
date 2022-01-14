#pragma once
#include <string>
#include <vector>


class DotGraph
{
public:
    DotGraph(bool directed);
    int AddNode(const std::string& label);
    void AddEdge(int nodeFrom, int nodeTo);
    std::string Build();
private:
    struct Node {
        int id;
        std::string label;
    };
    struct Edge {
        int nodeFrom;
        int nodeTo;
    };

    bool m_directed;
    std::vector<Node> m_nodes;
    std::vector<Edge> m_edges;
};