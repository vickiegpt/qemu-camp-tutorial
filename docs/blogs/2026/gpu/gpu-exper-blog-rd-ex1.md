# GPU 进阶实验 1 - 最小 AI 软件栈

!!! note "主要贡献者"

    - 作者：[@random25160765-collab](https://github.com/random25160765-collab)

进阶实验的内容非常多，因此按照主题分 p 进行分享。

---

首先来看实验 1。题目要求实现一套类 CUDA 风格的最小 AI 软件栈。我设定的目标是：在 QEMU 里面跑通一个 CNN。

在开始之前得先还基础实验阶段 built-in core 的技术债。首先用联合体简化寄存器读取逻辑：

```c
typedef union {
    uint32_t u32;
    int32_t i32;
    float f32;
    struct {
        uint32_t : 16;
        uint32_t bf16 : 16;
    };
    struct {
        uint32_t : 24;
        uint32_t e4m3 : 8;
    };
    struct {
        uint32_t : 24;
        uint32_t e5m2 : 8;
    };
    struct {
        uint32_t : 28;
        uint32_t e2m1 : 4;
    };
} GPURegister;

typedef struct GPGPULane {
    GPURegister gpr[GPGPU_NUM_REGS];   /* 通用寄存器 x0-x31 */
    GPURegister fpr[GPGPU_NUM_FREGS];  /* 浮点寄存器 f0-f31 */
    uint32_t pc;                     /* 程序计数器 */
    uint32_t mhartid;                /* 完整 hart ID (block|warp|lane) */
    uint32_t fcsr;                   /* fflags[4:0] | frm[7:5] */
    float_status fp_status;          /* softfloat 运行状态 */
    bool active;                     /* 是否活跃 */
} GPGPULane;
```

现在寄存器读取的书写就舒服多了！直接定义一组宏：

```c
/* ======== Register ======== */

/* default */
#define G(i)        (l->gpr[ctx->i].u32)
#define F(i)        (l->fpr[ctx->i].u32)
/* other choice */
#define G_CF(i)     (l->gpr[ctx->i].f32)
#define G_I32(i)    (l->gpr[ctx->i].i32)
#define F_CF(i)     (l->fpr[ctx->i].f32)
#define F_I32(i)    (l->fpr[ctx->i].i32)
#define F_BF16(i)   (l->fpr[ctx->i].bf16)
#define F_E4M3(i)   (l->fpr[ctx->i].e4m3)
#define F_E5M2(i)   (l->fpr[ctx->i].e5m2)
#define F_E2M1(i)   (l->fpr[ctx->i].e2m1)
```

接下来简化上下文。整数指令和浮点数指令的上下文分开来，特殊情况直接绕开宏写完整函数。

```c
/* ============== Context ================= */

/* context for int inst */
#define INIT_LANE_CONTEXT_IN() \
    GPGPULane *l = &ctx->warp->lanes[lane_id]; \
    uint32_t old_pc = l->pc; \
    uint32_t src1 = G(rs1); \
    uint32_t src2 = G(rs2); \
    int32_t imm = ctx->imm; \
    int32_t rd = ctx->rd; \
    (void)src1; (void)src2; (void)imm; (void)rd;

/* context for float inst */
#define INIT_LANE_CONTEXT_FP() \
    GPGPULane *l = &ctx->warp->lanes[lane_id]; \
    uint32_t old_pc = l->pc; \
    float32 src1 = F(rs1); \
    float32 src2 = F(rs2); \
    float32 src3 = F(rs3); \
    int32_t imm = ctx->imm; \
    int32_t rd = ctx->rd; \
    (void)src1; (void)src2; (void)src3; (void)imm; (void)rd;

/* 判断 pc 是否跳转 */
#define FINISH_LANE_CONTEXT() \
    if (l->pc == old_pc) l->pc += 4;
```

再定义不同的回调函数构造宏：

```c
/* ================ Func Wrapper ================ */

/* for int inst */
#define EXEC_FUNC_IN(name, code) \
    static void __attribute__((unused)) exec_##name(exec_ctx_t *ctx, int lane_id) { \
        INIT_LANE_CONTEXT_IN(); \
        code \
        FINISH_LANE_CONTEXT(); \
    }

/* for float inst */
#define EXEC_FUNC_FP(name, code) \
    static void __attribute__((unused)) exec_##name(exec_ctx_t *ctx, int lane_id) { \
        INIT_LANE_CONTEXT_FP(); \
        /* sync_fcsr_to_fp_status... */ \
        code \
        /* sync_fp_status_to_fcsr... */ \
        FINISH_LANE_CONTEXT(); \
    }
```

这样就彻底解决了回调函数臃肿的问题，代码更简洁，后续维护也方便多了。

---

还完技术债了，现在我们有了一个完整的 RV32IMF 解释器。来梳理一下思路：

我现在只有一个 QEMU 里面的虚拟 GPU，要在这个虚拟 GPU 上跑通 CNN，我需要做以下工作：
1. 配置好 host 和 guest 的环境
2. 定义设备接口，明确并组装控制链路
3. 准备好算子（乘法，加法等）设计顶层 api 并最终组装

首先要准备一个客户机，guest 选择预编译的 Debian RISC-V 镜像，然后在里面把基本的开发工具链装好；这个实验要用到 python，因此 python 相关的工具链也要装。然后要准备 host 上的开发环境，主要是 riscv 的交叉编译工具链，其他的到时候再说；然后要在 QEMU 里启动客户机。需要明确启动参数并准备好启动脚本。客户机在 QEMU 里启动成功后就可以进行下一步开发。

下一步需要写一个驱动，让它发现并正确控制虚拟 GPU 设备，并向上暴露更简单的接口。写完驱动之后可以设计一下怎么让 built-in core 进行运算，最终敲定的方案是使用手写的汇编算子，驱动把编译好的算子发送至 VRAM 让内核进行计算。

这一步踩了不少坑。客户机的架构是 rv64，但是后端只支持 rv32，内核无法识别客户机编译出来的指令，因此需要强制指定编译器生成 rv32 的指令。加上两个选项：`-march=rv32g -mabi=ilp32` 就好。

```bash
# 编译 kernel (以 maxpool 为例)
riscv64-linux-gnu-gcc -c -march=rv32g -mabi=ilp32 -O2 -nostdlib -ffreestanding maxpool.S -o maxpool.o
riscv64-linux-gnu-ld -T kernel.ld maxpool.o -o maxpool.elf
riscv64-linux-gnu-objcopy -O binary maxpool.elf maxpool.bin
```

kernel.ld 是链接脚本。在链接脚本中强制指定生成 32 位 elf 文件。

```text
OUTPUT_FORMAT("elf32-littleriscv", "elf32-littleriscv", "elf32-littleriscv")
OUTPUT_ARCH(riscv)
```

还有一种方式是在链接选项里显式添加 `-m elf32lriscv`，这样也可以。

设备尚未被 qtest 覆盖的部分有可能存在 bug，为了快速定位 bug，我建立了一个三级的日志体系，便于快速追踪 bug 的位置：

!!! note "日志系统"

    - 日志使用 tab 分离层级，最终形成树状的效果。
	
	```test
	[DEVICE] XXXX
		[KERNEL] XXXX
			[{name of inst}] XXXX
	```

	- 日志系统有两种粒度的控制。一种是在编译时传入标志，控制“是否加入日志代码”。一种是在运行时控制日志输出的等级，通过 mmio 控制。这种调试手段十分无脑，十分 agent-friendly。日志代码可以让 ai 生成，ai 也可以快速扫描并定位大量日志流中的 bug。
	
	- 前期我并未对日志输出进行任何控制，相当于永远输出全量日志，对于快速解决 bug 这是有好处的。但后期运行 matmul 甚至 cnn 的时候，指令级日志会严重干扰运行，io 开销会大大影响程序性能，耗时大约会增加数百倍。这个问题在调试 PoCL 运行时库的时候再次出现，不过那是后话了。

---

我让 ai 根据设备上下文写了一个最简驱动，只包含设备配置和基本操作。接下来就是写各种算子（向量加/乘，矩阵加/乘/乘加，ReLU，卷积，池化 etc.），编译完通过驱动发送给设备执行，然后找 bug，可能是驱动的 bug，也可能是设备的 bug，一边测试一遍修，慢慢地就把整个执行链路打通了。

这里面最令人头痛的就是传参问题了。grid/block/thread 相关的参数怎么获取，在哪里获取，需要明确规定；算子的参数也经常互相覆盖。因此需要明确 VRAM 的地址空间分配。

```c
/**
 * struct gpgpu_kernel_params - 内核启动参数
 */
struct gpgpu_kernel_params {
    __u32 grid_dim[3];   /* Grid 维度 (x, y, z) */
    __u32 block_dim[3];  /* Block 维度 (x, y, z) */
    __u64 kernel_addr;   /* 内核代码在 VRAM 中的地址 */
    __u64 args_addr;     /* 参数在 VRAM 中的地址 */
    __u32 shared_mem;    /* 共享内存大小 */
};
```

最后和 agent 确定下来，kernel 执行之前，先把参数传入寄存器；在 kernel 内部，参数通过一个越界的 VRAM 地址获取，即通过 VRAM 0x80000000 + offset 传递。core 会把这个越界地址转换为读 mmio 请求。另外规定输入数据在 VRAM 0x100000，权重在 0x200000，输出在 0x300000。

整个过程一个 ioctl 解决：

```c
/* 一次性设置并启动内核 (输入：struct gpgpu_kernel_params) */
#define GPGPU_IOCTL_LAUNCH_PARAMS   _IOW(GPGPU_IOC_MAGIC, 8, struct gpgpu_kernel_params)
```

后续对接 PoCL 时，尽管后端换成了 SimX，但是最后执行的还是 LLVM 编译出来的二进制代码，依然需要考虑 VRAM 的地址空间问题。

---

算子写好，在驱动层面通过所有测试之后，就要考虑用这些零件继续抽象和封装，做成类似 CUDA 那样的运行时库。需要实现的功能有：设备的初始化/重置/启动，malloc/free/memcpy/memset 等内存操作，内核执行。

出于“最简”的考量，内存管理我并没有采取复杂的手段，仅仅是简单的线性分配。整个“运行时库”几乎只有纯粹的转发代码，没有任何复杂的逻辑。同样，python 绑定也十分简单，整个过程基本没碰到逻辑上的阻碍。唯一的小麻烦是编译太慢。

python 绑定这一层用的是 pybind11，把上一层 cuda-like 的 api 包装了一下。最后的效果：

```python
import gpgpu, numpy as np

dev = gpgpu.Device()
a = gpgpu.Tensor(np.array([1,2,3,4], dtype=np.float32), dev)
b = gpgpu.Tensor(np.array([5,6,7,8], dtype=np.float32), dev)
c = a + b

print(c.numpy())   # [6. 8. 10. 12.]
```

看上去还比较像样。老实说，许多细节都有再打磨的空间，毕竟是一天 vibe 出来的结果。

最终测试 CNN 的时候碰到了大量问题。首先是算子行为与预期不符，经查是 agent 在编写算子时并未严格按照规则进行参数获取，把这个约束写入 memory 即可。其次是全量日志带来的性能问题；总共 2w 左右的参数，产生了数百万条指令级日志，跑了整整 10 分钟！而关闭日志之后只花了 1 秒。

测试用的 CNN 结构：

```text
网络结构（LeNet 变体，输入 32×32 单通道图像，10 类分类）：

  Input    (1, 32, 32)
    │
  Conv1    5×5，1→4 通道，无 padding    → (4, 28, 28)
  ReLU
  MaxPool  2×2                          → (4, 14, 14)
    │
  Conv2    3×3，4→8 通道，无 padding    → (8, 12, 12)
  ReLU
  MaxPool  2×2                          → (8, 6, 6)
    │
  Flatten                               → (288,)
    │
  FC1      288 → 64                     → (64,)
  ReLU
    │
  FC2      64 → 10                      → (10,)
  Softmax                               → (10,)  ∑=1
```

值得一提的是，guest 的编译效率非常低，尤其是用于 python 绑定的 cpp 文件，在 guest 上编译极慢（一次 5 分钟），于是我想在 host 上进行交叉编译。

kernel，runtime 等模块直接用交叉编译器就可以，而 pybind 文件的编译则需要 riscv 架构的 python 头文件。这部分文件需要从 guest 上裁剪，还是相当的麻烦，但比起等编译而言这点麻烦还是可以忍受的。后续对接 PoCL 时，需要在 guest 里面部署 PoCL 以及它的一大堆依赖，包括 LLVM。在 guest 里面编译这些是完全不可忍受的，必须交叉编译。

#### 附录

该阶段的项目结构：

```text
.
├── gpgpu.c
├── gpgpu.h
├── gpgpu_core.c
├── gpgpu_core.h
├── lpfp.h
├── inst.h
├── utils.h
├── memory.h
├── Makefile
├── guest/
│   ├── driver/                    # Linux 内核驱动
│   │   ├── Makefile
│   │   ├── gpgpu.c
│   │   ├── gpgpu.h
│   │   └── gpgpu_ioctl.h
│   ├── kernels/                   # GPU Kernels (汇编)
│   │   ├── Makefile
│   │   ├── kernel.ld              # 链接文件
│   │   ├── relu.S
│   │   ├── maxpool.S
│   │   ├── conv2d.S
│   │   ├── matmul.S
│   │   ├── softmax_exp.S
│   │   └── vecadd.S
│   ├── runtime/                   # CUDA-like 运行时
│   │   ├── Makefile
│   │   ├── gpgpu_runtime.h
│   │   └── gpgpu_runtime.c
│   └── python/                    # Python API
│       ├── gpgpu/
│       │   ├── __init__.py
│       │   ├── tensor.py
│       │   ├── nn.py
│       │   ├── functional.py
│       │   ├── profiler.py
│       │   └── runtime.py
│       ├── examples/
│       │   ├── cnn.py
│       │   ├── benchmark.py
│       │   └── test_operators.py
│       └── setup.py
```

该阶段的项目架构图：

```text
┌──────────────────────────────────────────────────────────────────┐
│                    Python API (gpgpu-torch)                 │
│  Tensor, nn.Module, nn.Conv2d, profiler                     │
│  用户最上层接口，PyTorch-like                                │
└────────────────────────────┬─────────────────────────────────────┘
                          │ ctypes / pybind11
┌────────────────────────────┴─────────────────────────────────────┐
│                   C Runtime (gpgpu_runtime)                │
│  gpgpuInit, gpgpuMalloc, gpgpuMemcpy, gpgpuLaunchKernel    │
│  CUDA-like API，管理设备、内存、Kernel 启动                  │
└────────────────────────────┬─────────────────────────────────────┘
                          │ ioctl
┌────────────────────────────┴─────────────────────────────────────┐
│                  Linux Kernel Driver (gpgpu.ko)            │
│  PCI probe/remove, BAR 映射，字符设备 /dev/gpgpu0,          │
│  mmap (VRAM), ioctl (LAUNCH_PARAMS, WAIT_KERNEL, ...),     │
│  中断处理 (轮询/传统 INTx)                                  │
└────────────────────────────┬─────────────────────────────────────┘
                          │ PCI 总线
┌────────────────────────────┴─────────────────────────────────────┐
│                  QEMU Virtual GPGPU Device                 │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  BAR0: MMIO 控制寄存器                                │   │
│  │  BAR2: VRAM (64MB)                                   │   │
│  │  BAR4: Doorbell 寄存器                               │   │
│  └───────────────────────────────────────────────────────────┘   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  RISC-V Built-in SIMT Core (RV32IMF)                 │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

未完待续。