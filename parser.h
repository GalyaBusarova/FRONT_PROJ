#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

#include "bin_reader.h"

// для расшифровки типа в onnx файле
enum DATA_TYPES
{
    UNDEFINED = 0,
    FLOAT = 1,
    UINT8 = 2,   
    INT8 = 3,    
    UINT16 = 4,  
    INT16 = 5,   
    INT32 = 6,   
    INT64 = 7,   
    STRING = 8,  
    BOOL = 9,    
    FLOAT16 = 10,
    DOUBLE = 11,
    UINT32 = 12,
    UINT64 = 13
};

// класс для хранения тензора
class Tensor
{
    std::string name;
    std::vector<int64_t> dims; // вектор размерностей
    int32_t data_type; // тип данных
    std::vector<uint8_t> raw_data;

public:
    std::string get_name()
    {
        return name;
    }

    void set_name(std::string tensor_name)
    {
        name = tensor_name;
    }

    void add_dim(int64_t dim)
    {
        dims.push_back(dim);
    }

    void set_data_type(int32_t type)
    {
        data_type = type;
    }

    void set_raw_data(const std::vector<uint8_t>& data) { raw_data = data; };
};

// класс, хранящий операцию и ее параметры
class Node
{
    std::string name;
    std::string op_type; // тип операции
    std::vector<std::string> inputs; // имена входных тензоров
    std::vector<std::string> outputs; // имена выходных тензоров

    // добавить атрибуты
public:
    void add_input(std::string input)
    {
        inputs.push_back(input);
    }

    void add_output(std::string output)
    {
        outputs.push_back(output);
    }

    void set_name(std::string node_name)
    {
        name = node_name;
    }

    void set_op_type(std::string op_name)
    {
        op_type = op_name;
    }

    const std::string& get_op_type() const { return op_type; }
    const std::vector<std::string>& get_inputs() const { return inputs; }
    const std::vector<std::string>& get_outputs() const { return outputs; }
    const std::string& get_name() const { return name; }
};

// класс для хранения графа
class Graph 
{
private:
    std::vector<Node> nodes;                          // все узлы
    std::unordered_map<std::string, Tensor> initializers;  // веса (поиск по имени)
    std::vector<std::string> inputs;                   // входы всей сети
    std::vector<std::string> outputs;                  // выходы всей сети

    int64_t ir_version;
    std::string producer_name;
    std::string producer_version;
    
    // Можно добавить имя графа
    std::string graph_name;

public:
    // сеттеры
    void setIrVersion(int64_t version) { ir_version = version; }
    void setProducerName(const std::string& name) { producer_name = name; }
    void setProducerVersion(const std::string& version) { producer_version = version; }
    void setGraphName(const std::string& name) {graph_name = name; }

    // добавить новую ноду
    void add_node(Node node)
    {
        nodes.push_back(node);
    }

    void add_tensor(Tensor tensor)
    {
        initializers[tensor.get_name()] = tensor;
    }

    // Возвращает список всех узлов (нужно для обхода графа и визуализации)
    const std::vector<Node>& get_nodes() const { return nodes; }

    // Возвращает веса/константы по имени (нужно для связывания входов узлов с весами)
    const std::unordered_map<std::string, Tensor>& get_initializers() const { return initializers; }

    // Возвращает имена входов всей сети (например, "input")
    const std::vector<std::string>& get_inputs() const { return inputs; }

    // Возвращает имена выходов всей сети (например, "output")
    const std::vector<std::string>& get_outputs() const { return outputs; }

    // Метаданные (нужно для отладки и тестов)
    int64_t getIrVersion() const { return ir_version; }
    const std::string& getProducerName() const { return producer_name; }
    const std::string& getProducerVersion() const { return producer_version; }
    const std::string& getGraphName() const { return graph_name; }
};


class ONNXParser 
{
private:
    BinaryReader reader;      // читает байты
    Graph graph;              // сюда собираем результат
    
    // вспомогательные методы
    void parseGraph(uint64_t length)
    {
        size_t end_pos = reader.get_cur_pos() + length;
        uint8_t cur_byte = reader.watch_cur_byte();

        while (reader.get_cur_pos() < end_pos) 
        {
            cur_byte = reader.watch_cur_byte();
            int wire_type = cur_byte & 0x07;
            int field_number = cur_byte >> 3;
            reader.read_byte();

            switch (field_number)
            {
                case 1: // Node, wire_type всегда равен 2
                {
                    uint64_t node_size = reader.read_varint(); // длина узла

                    Node res_node = parseNode(node_size);
                    graph.add_node(res_node);
                    break;
                }

                case 2: // name (graph name)
                {
                    uint64_t str_size = reader.read_varint();
                    auto bytes = reader.read_bytes(str_size);
                    std::string name(bytes.begin(), bytes.end());
                    graph.setGraphName(name);
                    break;
                }

                case 5: // initializer
                {
                    uint64_t tensor_size = reader.read_varint();

                    Tensor res_tensor = parseTensor(tensor_size);
                    graph.add_tensor(res_tensor);
                    break;
                }

                 
            }
        }
    }    

    // разбирает один узел 
    Node parseNode(uint64_t node_size) 
    {
        Node result;
        size_t end_pos = reader.get_cur_pos() + node_size;
        uint8_t cur_byte = reader.watch_cur_byte();
    
        while (reader.get_cur_pos() < end_pos) 
        {
            cur_byte = reader.watch_cur_byte();
            int wire_type = cur_byte & 0x07;
            int field_number = cur_byte >> 3;
            reader.read_byte();

            switch(field_number)
            {
                case 1: { // input (repeated string)
                    uint64_t str_size = reader.read_varint();  // длина очередной строки
                    std::vector<uint8_t> str_bytes = reader.read_bytes(str_size);
    
                    std::string input_name;
                    for (size_t i = 0; i < str_bytes.size(); i++) 
                    {
                        input_name += str_bytes[i];
                    }
    
                    result.add_input(input_name);  // добавляем в вектор inputs
                    break;
                }

                case 2: // output (repeated string)
                {
                    uint64_t str_size = reader.read_varint();  // длина очередной строки
                    std::vector<uint8_t> str_bytes = reader.read_bytes(str_size);
    
                    std::string output_name;
                    for (size_t i = 0; i < str_bytes.size(); i++) 
                    {
                        output_name += str_bytes[i];
                    }
    
                    result.add_output(output_name);  // добавляем в вектор inputs
                    break;
                }

                case 3: // name
                {
                    uint64_t str_size = reader.read_varint();

                    std::vector<uint8_t> str_symbols = reader.read_bytes(str_size);
                    std::string node_name;

                    for (size_t i = 0; i < str_symbols.size(); i++)
                    {
                        char symbol = str_symbols[i];
                        node_name += symbol;
                    }

                    std::cout << node_name;

                    result.set_name(node_name);
                    break;
                }

                case 4: // op_type
                { 
                    uint64_t str_size = reader.read_varint();  // длина строки
                    std::vector<uint8_t> str_bytes = reader.read_bytes(str_size);
    
                    std::string op_type;
                    for (size_t i = 0; i < str_bytes.size(); i++) {
                        op_type += str_bytes[i];  // собираем символы
                    }
    
                    result.set_op_type(op_type);  // сохраняем в Node
                    std::cout << "op_type: " << op_type << std::endl;  // для отладки
                    break;
                }

                case 5: // attribute
                {
                    uint64_t attr_len = reader.read_varint();  // длина атрибута
                    // Просто перепрыгиваем эти байты
                    reader.read_bytes(attr_len);
                    break;
                } 
            }
        // читаете ключ
        // switch по field_number:
        // 1: inputs (repeated string)
        // 2: outputs (repeated string)
        // 3: name (string)
        // 4: op_type (string) ← самое важное!
        // 5: attribute (сложно) ← пока можно пропустить
        }

        return result;  
    }

    // разбирает один тензор
    Tensor parseTensor(uint64_t tensor_size)
    {
        Tensor result;

        size_t end_pos = reader.get_cur_pos() + tensor_size;
        uint8_t cur_byte = reader.watch_cur_byte();

        while (reader.get_cur_pos() < end_pos)
        {
            cur_byte = reader.watch_cur_byte();
            int wire_type = cur_byte & 0x07;
            int field_number = cur_byte >> 3;
            reader.read_byte();

            switch(field_number)
            {
                case 1: // dims
                {
                    uint64_t dim = reader.read_varint(); 
                    result.add_dim(dim);
                    break;
                }

                case 2: // data_type
                {
                    uint64_t type = reader.read_varint(); 
                    result.set_data_type(type);
                    break;
                }

                case 3: // name
                {
                    uint64_t str_size = reader.read_varint();

                    std::vector<uint8_t> str_symbols = reader.read_bytes(str_size);
                    std::string tensor_name;

                    for (size_t i = 0; i < str_symbols.size(); i++)
                    {
                        char symbol = str_symbols[i];
                        tensor_name += symbol;
                    }

                    std::cout << tensor_name;

                    result.set_name(tensor_name);
                    break;
                }

                case 5: // float_data
                {
                    uint64_t len = reader.read_varint();
                    std::vector<uint8_t> bytes = reader.read_bytes(len);
                    
                    result.set_raw_data(bytes);  
                    break;
                }

                case 9: // raw_data
                {
                    uint64_t len = reader.read_varint();
                    result.set_raw_data(reader.read_bytes(len));

                    break;
                }
            }
        }

        return result;
    }                  
    
public:
    ONNXParser(const std::string& filename);
    Graph parse();            // публичный метод — только он вызывается снаружи
};

ONNXParser::ONNXParser(const std::string& filename) 
    : reader(filename)
{
}

# if 0
Graph ONNXParser::parse()
{
    Graph result;

    while (!reader.check_eof())
    {
        uint8_t cur_byte = reader.watch_cur_byte();

        int wire_type = cur_byte & 0x07;
        int field_number = cur_byte >> 3;

        reader.read_byte();

        switch (field_number)
        {
            case 1: // ir_version
                if (wire_type == 0) // VARINT
                {
                    uint64_t ir_version = reader.read_varint();
                    std::cout << ir_version;

                    result.setIrVersion(ir_version);
                }
                break;

            case 2: // producer_name
                if (wire_type == 2) // LEN
                {
                    uint64_t str_size = reader.read_varint();

                    std::vector<uint8_t> str_symbols = reader.read_bytes(str_size);
                    std::string prod_name;

                    for (size_t i = 0; i < str_symbols.size(); i++)
                    {
                        char symbol = str_symbols[i];
                        prod_name += symbol;
                    }

                    std::cout << prod_name;

                    result.setProducerName(prod_name);
                }
                break;

            case 3: // producer version
                if (wire_type == 2) // LEN
                {
                    uint64_t str_size = reader.read_varint();

                    std::vector<uint8_t> str_symbols = reader.read_bytes(str_size);
                    std::string prod_version;

                    for (size_t i = 0; i < str_symbols.size(); i++)
                    {
                        char symbol = str_symbols[i];
                        prod_version += symbol;
                    }

                    std::cout << prod_version;

                    result.setProducerVersion(prod_version);
                }
                break;

            case 7: // graph
                if (wire_type == 2) // LEN
                {
                    uint64_t graph_size = reader.read_varint();

                    parseGraph(graph_size);
                }
                break;
        }
    }

    ONNXParser::graph = result;

    return result;
}
#endif 

Graph ONNXParser::parse()
{
    while (!reader.check_eof())
    {
        try {
            uint8_t cur_byte = reader.watch_cur_byte();
            int wire_type = cur_byte & 0x07;
            int field_number = cur_byte >> 3;
            reader.read_byte();

            switch (field_number) {
                case 1: // ir_version
                    if (wire_type == 0) graph.setIrVersion(reader.read_varint());
                    break;

                # if 0
                case 2: // producer_name
                    if (wire_type == 2) {
                        uint64_t len = reader.read_varint();
                        graph.setProducerName(std::string(reader.read_bytes(len).begin(), 
                                                          reader.read_bytes(len).end()));
                    }
                    break;
                #endif 

                case 2: // producer_name
                    if (wire_type == 2) 
                    {
                        uint64_t len = reader.read_varint();
                        auto bytes = reader.read_bytes(len);  
                        graph.setProducerName(std::string(bytes.begin(), bytes.end()));
                    }
                    break;


                case 7: // graph
                    if (wire_type == 2) {
                        uint64_t graph_size = reader.read_varint();
                        parseGraph(graph_size);
                    }
                    break;
                default:
                    // Пропуск неизвестных полей
                    if (wire_type == 0) reader.read_varint();
                    else if (wire_type == 1) reader.read_bytes(8);
                    else if (wire_type == 2) { uint64_t l = reader.read_varint(); reader.read_bytes(l); }
                    else if (wire_type == 5) reader.read_bytes(4);
                    break;
            }
        } catch (const std::out_of_range&) {
            // Достигли конца файла — это нормально, выходим
            break;
        } catch (const std::runtime_error& e) {
            // Реальная ошибка парсинга
            std::cerr << "Parse error: " << e.what() << "\n";
            throw;
        }
    }

    return graph;
}