# ONNX Parser

Парсер вычислительных графов нейронных сетей из формата **ONNX** на языке **C++**.

Проект реализует структуру данных граф, преобразование ONNX модели в внутренний граф, поддержку основных операций нейронных сетей и визуализацию с помощью GraphViz.

---

## Возможности

| Функция | Описание |
|---------|----------|
| **Парсинг ONNX** | Чтение бинарного формата (protobuf/varint) |
| **Граф на C++** | Классы `Graph`, `Node`, `Tensor` |
| **8+ операций** | Conv, Relu, Gemm, MatMul, Add, Mul, Reshape, Concat |
| **Атрибуты** | strides, dilations, group, alpha, beta, transB, allowzero, auto_pad |
| **Визуализация** | Экспорт в GraphViz DOT с цветами и формами |
| **Тесты** | 3 тестовые модели + CMake testing |

---

## Требования

| Компонент | Версия | Зачем |
|-----------|--------|-------|
| C++ компилятор | C++17 (GCC 7+, Clang 5+, AppleClang 15+) | Сборка проекта |
| CMake | 3.10+ | Система сборки |
| GraphViz | Любая  | Визуализация графа |

## Cборка

```bash
# 1. Создайте папку build
mkdir -p build && cd build

# 2. Запустите CMake
cmake ..

# 3. Соберите проект
make

# 4. Запустите тесты
ctest --verbose
```
## Запуск

```bash
# Базовый запуск
./parser path/to/model.onnx

# Примеры с тестовыми моделями
./parser ../tests/simple_matmul.onnx
./parser ../tests/complex_net.onnx
./parser ../tests/custom_net.onnx
```

### Пример вывода
```bash
=== Loading: tests/complex_net.onnx ===

=== Parsed Graph Info ===
IR version: 10
Producer: pytorch v2.10.0
Graph name: main_graph

=== Nodes ===
Op: Conv
  Inputs: input conv1.weight conv1.bias
  Outputs: conv2d
  [group: 1]
  [strides: 1 1]
  [dilations: 1 1]

Op: Relu
  Inputs: conv2d
  Outputs: relu

...

✅ Parsing completed successfully!
```

## Поддерживаемые операции и их атрибуты

| Операция | Атрибуты | Описание |
|----------|----------|----------|
| **Conv** | `strides`, `dilations`, `group`, `auto_pad` | Свертка |
| **Relu** | — | Функция активации |
| **Gemm** | `alpha`, `beta`, `transA`, `transB` | Полносвязный слой |
| **MatMul** | — | Умножение матриц |
| **Add** | — | Поэлементное сложение |
| **Mul** | — | Поэлементное умножение |
| **Reshape** | `allowzero` | Изменение формы тензора |
| **Concat** | `axis` | Конкатенация тензоров |
| **Shape** | — | Получение формы тензора |


## Структура проекта
```bash
.
├── CMakeLists.txt          # Конфигурация сборки
├── README.md               # Документация
├── .gitignore              # Игнорируемые файлы
├── include/
│   ├── bin_reader.h        # Чтение байтов и varint
│   └── parser.h            # Классы Graph, Node, Tensor
├── src/
│   ├── main.cpp            # Точка входа
│   └── parser.cpp          # Реализация парсера
└── tests/
    ├── simple_matmul.onnx  # Тест 1: Базовый MatMul
    ├── complex_net.onnx    # Тест 2: CNN + FC слои
    └── custom_net.onnx     # Тест 3: Реальная модель
```

## Тесты
Проект включает 3 тестовые модели:
| Модель | Описание | Операции |
|--------|----------|----------|
| `simple_matmul.onnx` | Простое умножение матриц | `MatMul` |
| `complex_net.onnx` | CNN + Fully Connected | `Conv`, `Relu`, `Reshape`, `Gemm` |
| `custom_net.onnx` | Реальная модель | `Conv`, `Relu`, `Add`, `Mul`, `Gemm` |

### Запуск тестов
```bash
cd build
ctest --verbose
```

## Архитектура
### Классы
| Класс | Описание |
|-------|----------|
| **BinaryReader** | Низкоуровневое чтение байтов и varint |
| **Tensor** | Хранение тензора (имя, размеры, тип, данные) |
| **Node** | Операция графа (тип, входы, выходы, атрибуты) |
| **Graph** | Вычислительный граф (узлы, тензоры, входы, выходы) |
| **ONNXParser** | Главный парсер (чтение ONNX → Graph) |

### Формат ONNX
ONNX использует protobuf сериализацию:
Varint — кодирование целых чисел переменной длины
Wire types — типы полей (0=varint, 2=length-delimited, 5=fixed32)
Field numbers — идентификаторы полей протокола

## Визуализация графа

Пример работы парсера на модели `custom_net.onnx`:

![Граф нейросети](images/graph.png)

## Автор

**Галина Бусарова**

Проект выполнен в рамках курса по тензорным компиляторам.
