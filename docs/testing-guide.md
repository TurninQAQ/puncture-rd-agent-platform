# Testing Guide

## 1. Test layers

### Contract tests

验证字段、枚举、工具名称、返回结构和错误码不被破坏。任何真实实现都必须首先通过。

### Unit tests

验证模块内部算法。由实现该模块的模型补充，例如几何读取、Chunk、RRF、射线追踪等。

### Failure tests

主动构造 spacing 不一致、缺标签、空 Mask、超时、无路径、版本冲突等场景，验证错误码和分支。

### Graph tests

验证给定工具结果后 Agent 是否走正确节点，而不是验证医学算法本身。

### Integration tests

使用全部 Mock 跑通端到端；每替换一个真实模块后重复执行。

### Regression tests

记录模型、Prompt、索引、工具和配置版本，比较任务成功率、工具参数准确率、安全回归和延迟。

## 2. Required test document for every module

每个 `specs/*.md` 必须写：

1. 正常输入测试。
2. 边界输入测试。
3. 非法输入测试。
4. 依赖服务失败测试。
5. 幂等/重复请求测试。
6. 性能测试。
7. Contract Test文件位置。
8. 验收标准。

## 3. Commands

当前无第三方依赖：

```bash
python3 run_tests.py
```

未来安装项目依赖后可以增加 pytest，但不得删除标准库 Contract Tests。

## 4. Evidence

测试不能只写“通过”。至少保存：测试集版本、输入fixture、预期输出、实际输出、模型/工具版本、耗时和失败日志。

