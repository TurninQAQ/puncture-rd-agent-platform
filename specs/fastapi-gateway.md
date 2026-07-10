# FastAPI Gateway and Run Runtime Specification

## 1. Goal and boundary

API Gateway 将调用方请求转换为固定的 Agent Run 契约，并提供状态查询、事件回放、审批、取消和恢复。它不直接调用医学算法，不把原始三维数组或内部 URI 放入 HTTP 响应。

Mock 阶段实现内存 service class 和 demo CLI；生产阶段再加 FastAPI、SSE、PostgreSQL checkpoint、鉴权和限流。两阶段必须共享同一请求/响应模型。

## 2. Required endpoints

- `POST /api/v1/runs`
- `GET /api/v1/runs/{run_id}`
- `GET /api/v1/runs/{run_id}/events`
- `POST /api/v1/runs/{run_id}/approvals/{approval_id}`
- `POST /api/v1/runs/{run_id}/cancel`
- `POST /api/v1/runs/{run_id}/resume`
- `GET /api/v1/artifacts/{artifact_id}/metadata`
- `GET /health`
- `GET /metrics`

## 3. Run lifecycle

```text
PENDING -> RUNNING -> SUCCEEDED
                  -> WAITING_APPROVAL -> RUNNING
                  -> FAILED
                  -> CANCELLED
```

规则：

- 终态 `SUCCEEDED/FAILED/CANCELLED` 不可恢复为运行态。
- `resume` 只能恢复具备持久化 checkpoint 的可恢复 Run。
- `approval` 必须匹配当前等待中的 `approval_id`，过期或重复审批应返回冲突。
- 同一租户、同一 idempotency key 的创建请求返回同一 `run_id`。
- 事件 sequence 从 1 单调递增；同一事件不得重复产生不同 payload。

## 4. Create-run validation

创建请求至少检查：

- `case_id` 非空；
- `task_type` 属于契约枚举；
- 输入 artifact ID 格式合法；
- request principal 对 case/artifact 有权限；
- `idempotency_key` 非空且长度受限；
- 约束参数在业务允许范围内；
- 请求体大小不超过限制且不包含像素数组。

验证失败返回固定错误结构，不创建 Run，也不调用 Agent。

## 5. Event contract

每个事件至少包含：

- `run_id`；
- `sequence`；
- `event_type`；
- `node_name`；
- `timestamp`；
- 脱敏后的 `payload`；
- `trace_id`。

事件类型至少覆盖 run created、node started、node completed、tool called、tool result、approval requested、run completed、run failed、run cancelled。

SSE 重连时使用 last sequence 游标，从下一条事件开始回放；不得跳序或把内部 URI/Prompt 全文泄漏给无权限调用方。

## 6. Checkpoint and recovery

生产实现每个具有外部副作用的节点完成后保存 checkpoint，至少包括：

- 固定 Agent State；
- 当前/下一节点；
- 已完成 tool call ID；
- 输出 artifact ID；
- graph、prompt、model、tool 版本；
- 最后 event sequence。

进程重启后从最后一个已提交 checkpoint 恢复。已完成且具备相同 idempotency key 的工具不得重复执行。

## 7. Authorization and privacy

- 所有 run、artifact、event 查询按 tenant/project/case 做权限校验。
- API 返回 artifact metadata，不返回内部 URI。
- trace 中患者标识、token、Authorization header 必须脱敏。
- 生产实现建议 OIDC/JWT；Mock 使用固定 principal/role 即可测试语义。

## 8. Mock implementation behavior

内存 service 必须支持：

1. 创建并同步执行一个 Mock Agent Run；
2. 按 ID 查询快照；
3. 按 sequence 返回有序事件；
4. 幂等创建；
5. 取消尚未终止的 Run；
6. 对等待审批的 Run 执行 approve/reject；
7. 返回结构化错误；
8. CLI 打印最终状态、报告摘要和事件数量。

Mock 不需要启动网络端口，也不得引入 FastAPI 依赖。

## 9. Correctness test matrix

| ID | Scenario | Expected result |
|---|---|---|
| API-001 | 合法创建请求 | 返回 run_id，状态最终符合 Mock 场景 |
| API-002 | 重复 idempotency key | 返回同一 run_id，不重复执行工具 |
| API-003 | 空 case_id | 结构化 INVALID_REQUEST，无 Run |
| API-004 | 未知 task_type | 结构化 INVALID_REQUEST |
| API-005 | 查询未知 run_id | NOT_FOUND |
| API-006 | 事件回放 | sequence 严格递增且与图转移一致 |
| API-007 | 使用 last sequence 重连 | 只返回游标之后的事件 |
| API-008 | 等待审批后 approve | 从 checkpoint 恢复并完成 |
| API-009 | 错误 approval_id | CONFLICT，状态不变 |
| API-010 | 重复审批 | 幂等或明确 CONFLICT，不重复执行节点 |
| API-011 | 运行中 cancel | 进入 CANCELLED，后续节点不执行 |
| API-012 | 对终态 cancel/resume | 返回冲突，终态不变 |
| API-013 | 未授权查询 run/artifact | FORBIDDEN，不泄漏是否存在的敏感细节 |
| API-014 | Agent dependency timeout | Run 进入 FAILED/可恢复状态并保存错误 |
| API-015 | 进程重启恢复 | 从最后 checkpoint 继续，已完成工具不重复执行 |
| API-016 | 响应与事件脱敏 | 无 internal_uri、token、原始影像内容 |

## 10. Performance and reliability tests

- 100 个并发 create/get 请求无状态串线。
- 单 Run 10,000 个事件分页/游标读取顺序正确。
- 重复 100 次 idempotent create 仅产生一个 Run。
- 模拟 checkpoint 数据库暂时失败，API 返回可重试错误，不谎报成功。
- 记录 create、get、event replay 的 p50/p95 和错误率；Mock 只建立基线，生产阈值按部署资源确定。

## 11. Test locations and commands

- Contract: `tests/contract/test_api_contracts.py`
- Unit: `tests/api/test_run_service.py`
- Integration: `tests/integration/test_mock_workflow.py`

```bash
python3 run_tests.py
python3 -m unittest discover -s tests/api -p 'test_*.py' -v
```

生产 FastAPI 实现还应增加 TestClient/HTTP/SSE/鉴权/重启恢复测试，但不得删除标准库测试。

## 12. Acceptance

- API-001 至 API-016 全部通过。
- Mock demo 可从创建请求得到最终报告和有序事件。
- 幂等、审批、取消、恢复、权限和脱敏行为均有可重复测试。
- 更换为 FastAPI 后公共契约和 Contract Tests 不变。
- 实现者提交 OpenAPI diff、测试结果、性能基线、失败注入结果和已知限制。
