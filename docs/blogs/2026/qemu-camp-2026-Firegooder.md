#   背景介绍

  我是一名电子信息专业的学生，对计算机体系结构有兴趣，之前学 XV6 课程的时候接触到了 QEMU，但是一直停留在最基础的使用阶段，看到 QEMU 训练营的招募后，我觉得这是一个难得的动手机会，于是报名参加了 GPGPU 方向的实验。

#   专业阶段

我选择了 GPGPU 设备建模 方向。实验任务是实现一个教育用途的 GPU 设备模型，通过 QTest 框架的 17 个测题来验证正确性。

  GPGPU 是一个 PCIe 设备，核心数据结构是 GPGPUState：

``` 
 struct GPGPUState {
      PCIDevice parent_obj;           // PCI 设备基类
      MemoryRegion ctrl_mmio;         // BAR0: 控制寄存器
      MemoryRegion vram;              // BAR2: 显存
      uint8_t *vram_ptr;              // 显存后端存储
      uint32_t global_ctrl;           // 设备使能/复位
      uint32_t global_status;         // 就绪/忙碌/错误
      GPGPUKernelParams kernel;       // 内核分发参数
      GPGPUSIMTContext simt;          // 线程上下文
  };
```

  关键理解：QEMU 的设备模型本质上是"维护状态 + 响应读写"。BAR 空间通过 MemoryRegionOps 关联回调函数，宿主机侧 QTest 通过 qpci_io_readl/writel 触发这些回调，直接读写 MMIO 寄存器——整个过程无需客户机操作系统参与。

  RISC-V SIMT 解释器的设计

  GPU 核心用一个精简的 RISC-V 解释器模拟。每个 warp 包含 32 个 lane，锁步执行同一条指令：

```
while (lane->active && cycles < max_cycles) {
      uint32_t inst = ldl_le_p(s->vram_ptr + lane->pc);
      switch (OPCODE(inst)) {
      case OPCODE_LUI:       lane->gpr[rd] = UTYPE_IMM(inst);         break;
      case OPCODE_STORE:     stl_le_p(vram + addr, rs2_val);          break;
      case OPCODE_OP_FP:     /* FADD.S / FMUL.S / narrow-convert */   break;
      case OPCODE_SYSTEM:    /* CSRRS(mhartid) / EBREAK */            break;
      }
      lane->pc += 4;
  }
```

  线程通过 mhartid CSR 获取自己的三维 ID：

  mhartid 位域： [31:13] block_id   [12:5] warp_id   [4:0] lane_id

  这相当于 CUDA 的 threadIdx + blockIdx 被编码进一个硬件寄存器，内核代码通过一条 csrrs 指令就能拿到自己的完整身份。

  内核分发流程的分发流程如下：

  测试 → 写 VRAM 上传内核代码 → 配置 grid/block 维度 → 写 DISPATCH 寄存器触发执行：

```
  BAR2 VRAM                 BAR0 控制寄存器
  ┌──────────────┐          ┌──────────────────┐
  │ 0x0000: code │          │ GRID_DIM  = 1x1x1│
  │ 0x1000: data │          │ BLOCK_DIM = 8x1x1│
  └──────────────┘          │ KERNEL_ADDR = 0  │
                            │ DISPATCH = 1 ────→ 触发 gpgpu_core_exec_kernel()
                            └──────────────────┘
```

#   总结

 最大收获是从零理解了 QEMU 设备模型的完整链路：PCI 设备注册 → BAR 空间映射 → MMIO 读写回调 → QOS
  测试框架自动发现和验证。以前看 QEMU 源码总觉得庞杂无从下手，现在至少有一条清晰的路径了。

