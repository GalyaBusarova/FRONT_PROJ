// main.cpp
#include <iostream>
#include <stdexcept>
#include "parser.h"  


int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>\n";
        return 1;
    }

    try {
        std::cout << "=== Loading: " << argv[1] << " ===\n";
        
        ONNXParser parser(argv[1]);
        Graph graph = parser.parse();
        
        std::cout << "\n=== Parsed Graph Info ===\n";
        std::cout << "IR version: " << graph.getIrVersion() << "\n";
        std::cout << "Producer: " << graph.getProducerName() 
                  << " v" << graph.getProducerVersion() << "\n";
        std::cout << "Graph name: " << graph.getGraphName() << "\n\n";
        
        // Выводим узлы (нужны геттеры в Graph/Node)
        // Если геттеров ещё нет — временно сделайте nodes/public или добавьте:
        // const std::vector<Node>& get_nodes() const { return nodes; }
        
        std::cout << "=== Nodes ===\n";
         for (const auto& node : graph.get_nodes()) {
             std::cout << "Op: " << node.get_op_type() << "\n";
             std::cout << "  Inputs: ";
             for (const auto& in : node.get_inputs()) std::cout << in << " ";
             std::cout << "\n  Outputs: ";
             for (const auto& out : node.get_outputs()) std::cout << out << " ";
             std::cout << "\n\n";
        }
        
        std::cout << "✅ Parsing completed successfully!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << "\n";
        return 1;
    }
}