# Terminology

- Artifact: 不可变的 CT、Mask、路径或报告文件及其元数据。
- Geometry fingerprint: 由 shape、spacing、origin、direction 和坐标系生成的几何哈希。
- Contract: 模块不得自行修改的输入输出和错误约定。
- Stub: 尚未实现、明确抛出 NotImplementedError 的占位实现。
- Mock: 返回固定可预测结果、用于联调和测试的模拟实现。
- Verifier: 不依赖 LLM 的确定性准入和一致性检查。
- Profile: 版本化的模型、标签、规划或风险配置。
- Tool result: 算法工具证据；与 RAG 文档 citation 分开。

