import torch
import torch.nn as nn
import torch.onnx
import onnx

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        
        # 1. Conv операция
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        
        # Второй сверточный слой: 4 -> 8 каналов
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)
        
        # ✅ ИСПРАВЛЕНО: Параметры для Mul/Add теперь с 8 каналами (как выход conv2)
        self.scale_param = nn.Parameter(torch.ones(8, 1, 1))   # Для Mul
        self.shift_param = nn.Parameter(torch.zeros(8, 1, 1))  # Для Add
        
        # Веса для ручного MatMul: вход 8*14*14=1568, выход 64
        self.matmul_weight = nn.Parameter(torch.randn(64, 1568))
        
        # Gemm через nn.Linear
        self.fc_gemm = nn.Linear(64, 10)
        self.fc_out = nn.Linear(10, 2)

    def forward(self, x):
        # Conv + Relu
        x = self.conv1(x)
        x = torch.relu(x)
        
        x = self.conv2(x)
        x = torch.relu(x)
        
        # ✅ Mul (явное умножение) - теперь размеры совпадают!
        x = x * self.scale_param
        
        # ✅ Add (явное сложение)
        x = x + self.shift_param
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # ✅ MatMul (ручное умножение матриц)
        x = torch.matmul(x, self.matmul_weight.t())
        x = torch.relu(x)
        
        # ✅ Gemm (через nn.Linear)
        x = self.fc_gemm(x)
        x = torch.relu(x)
        x = self.fc_out(x)
        
        return x

# Создание модели
model = CustomNet()
model.eval()

# Dummy input
dummy_input = torch.randn(1, 1, 28, 28)

# ✅ Экспорт с поддержкой нового экспортера PyTorch
torch.onnx.export(
    model,
    dummy_input,
    "custom_net.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=17,  # Используем более новый opset
    do_constant_folding=False,
    # ✅ Отключаем строгий режим нового экспортера, если он вызывает проблемы
    dynamic_axes=None,
)

print("✅ Модель сохранена в custom_net.onnx")

# Проверка
try:
    onnx_model = onnx.load("custom_net.onnx")
    onnx.checker.check_model(onnx_model)
    print("✅ Модель успешно проверена!\n")
    
    # Статистика операций
    op_counts = {}
    for node in onnx_model.graph.node:
        op = node.op_type
        op_counts[op] = op_counts.get(op, 0) + 1
    
    print("📊 Операции в модели:")
    required = ['Add', 'Mul', 'Conv', 'Relu', 'MatMul', 'Gemm']
    for op in required:
        cnt = op_counts.get(op, 0)
        mark = "✅" if cnt > 0 else "❌"
        print(f"  {mark} {op}: {cnt}")
        
except Exception as e:
    print(f"❌ Ошибка: {e}")