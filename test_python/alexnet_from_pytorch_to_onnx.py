import torch
import torchvision
import onnx
# import onnxruntime

dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
model = torchvision.models.alexnet(pretrained = True).cuda()
print(type(model))
print("model: {}".format(model))
input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
print(input_names)
output_names = ["output1"]
# torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)

model = onnx.load("alexnet.onnx")
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))
