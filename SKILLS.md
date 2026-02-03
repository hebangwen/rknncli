## 简单解析 .rknn 文件

.rknn 的文件格式为：

1. `RKNN`：4 byte，文件开头 magic number
2. 填充：4 byte，使用 0 填充
3. 文件格式：8 byte，目前文件格式一般为 6
4. 文件长度：8 byte
5. 文件头填充：40 byte，如果文件格式大于 1，则文件头填充不存在，为 0 byte
6. 实际 RKNN 模型数据：由文件长度决定
7. 末尾 JSON 字符串长度：8 byte
8. 末尾 JSON 字符串

文件头一共 24 字节或者 64 字节，由 RKNN version 决定。得到 JSON 字符串长度和对应的偏移量之后，就可以得到用于标识模型信息的 json 字符串。

一次分析流程如下：

```bash

# 1. 得到文件长度为 8357380
-> % ls -l assets/yolov5s-640-640.rknn 
-rw-r--r--@ 1 hebangwen  staff  8357380 Feb  2 22:13 assets/yolov5s-640-640.rknn

# 2. 得到文件长度为 0x7f7f40 = 8355648
-> % hexdump -C -v -n 64 assets/yolov5s-640-640.rknn 
00000000  52 4b 4e 4e 00 00 00 00  06 00 00 00 00 00 00 00  |RKNN............|
00000010  40 7f 7f 00 00 00 00 00  00 00 00 00 00 00 00 00  |@...............|
00000020  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00000030  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00000040

# 3. 打印模型信息
-> % dd if=assets/yolov5s-640-640.rknn bs=1 skip=$((8355648+64+8)) count=$((8357380-8355648-64-8)) status=none | jq
{
  "connection": [
    {
      "left": "input",
      "left_tensor_id": 0,
      "node_id": 0,
      "right_tensor": {
        "tensor_id": 0,
        "type": "norm_tensor"
      }
    },
    {
      "left": "output",
      "left_tensor_id": 0,
      "node_id": 0,
      "right_tensor": {
        "tensor_id": 1,
        "type": "norm_tensor"
      }
    },
    {
      "left": "output",
      "left_tensor_id": 1,
      "node_id": 0,
      "right_tensor": {
        "tensor_id": 2,
        "type": "norm_tensor"
      }
    },
    {
      "left": "output",
      "left_tensor_id": 2,
      "node_id": 0,
      "right_tensor": {
        "tensor_id": 3,
        "type": "norm_tensor"
      }
    }
  ],
  "const_tensor": [],
  "graph": [
    {
      "left": "input",
      "left_tensor_id": 0,
      "right": "norm_tensor",
      "right_tensor_id": 0
    },
    {
      "left": "output",
      "left_tensor_id": 0,
      "right": "norm_tensor",
      "right_tensor_id": 1
    },
    {
      "left": "output",
      "left_tensor_id": 1,
      "right": "norm_tensor",
      "right_tensor_id": 2
    },
    {
      "left": "output",
      "left_tensor_id": 2,
      "right": "norm_tensor",
      "right_tensor_id": 3
    }
  ],
  "input_num": 1,
  "name": "rknn model",
  "network_platform": "ONNX",
  "node_num": 1,
  "nodes": [
    {
      "input_num": 1,
      "lid": "npu_network_bin_graph",
      "name": "nnbg",
      "nn": {
        "nbg": {
          "type": "RKNN_OP_NNBG"
        }
      },
      "op": "RKNN_OP_NNBG",
      "output_num": 3,
      "uid": 0
    }
  ],
  "norm_tensor": [
    {
      "dim_num": 4,
      "dtype": {
        "qnt_method": "layer",
        "qnt_type": "int8",
        "vx_type": "int8"
      },
      "size": [
        1,
        3,
        640,
        640
      ],
      "tensor_id": 0,
      "url": "images"
    },
    {
      "dim_num": 4,
      "dtype": {
        "qnt_method": "layer",
        "qnt_type": "int8",
        "vx_type": "int8"
      },
      "size": [
        1,
        255,
        80,
        80
      ],
      "tensor_id": 1,
      "url": "output0"
    },
    {
      "dim_num": 4,
      "dtype": {
        "qnt_method": "layer",
        "qnt_type": "int8",
        "vx_type": "int8"
      },
      "size": [
        1,
        255,
        40,
        40
      ],
      "tensor_id": 2,
      "url": "286"
    },
    {
      "dim_num": 4,
      "dtype": {
        "qnt_method": "layer",
        "qnt_type": "int8",
        "vx_type": "int8"
      },
      "size": [
        1,
        255,
        20,
        20
      ],
      "tensor_id": 3,
      "url": "288"
    }
  ],
  "norm_tensor_num": 4,
  "ori_network_platform": "ONNX",
  "output_num": 3,
  "target_platform": [
    "rk3588"
  ],
  "version": "1.6.2-source_code",
  "virtual_tensor": []
}

```

---
